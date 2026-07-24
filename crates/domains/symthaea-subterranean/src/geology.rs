// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic geotechnical profiles for subterranean simulation.
//!
//! Depth alone is not geology. A credible underground platform needs explicit
//! strata whose hardness, permeability, gas potential, roof cohesion, ore
//! grade, and survey confidence drive the plant and its hazards.

use serde::{Deserialize, Serialize};

pub const MAX_SIMULATED_DEPTH_M: f64 = 200.0;
const CONTIGUITY_EPSILON_M: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialClass {
    Unconsolidated,
    Clay,
    Sandstone,
    Limestone,
    Shale,
    Granite,
    OreBody,
    FaultGouge,
}

impl MaterialClass {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unconsolidated => "unconsolidated",
            Self::Clay => "clay",
            Self::Sandstone => "sandstone",
            Self::Limestone => "limestone",
            Self::Shale => "shale",
            Self::Granite => "granite",
            Self::OreBody => "ore_body",
            Self::FaultGouge => "fault_gouge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stratum {
    pub top_depth_m: f64,
    pub bottom_depth_m: f64,
    pub material: MaterialClass,
    /// Relative resistance to cutter penetration, in [0, 1].
    pub hardness: f64,
    /// Relative cutter wear imposed per unit penetration, in [0, 1].
    pub abrasiveness: f64,
    /// Relative fluid transmissivity, in [0, 1].
    pub permeability: f64,
    /// Relative probability of hazardous gas release, in [0, 1].
    pub gas_potential: f64,
    /// Unsupported roof competence, in [0, 1].
    pub roof_cohesion: f64,
    /// Relative target mineral grade, in [0, 1].
    pub ore_grade: f64,
    /// Confidence in this stratum model, in [0, 1].
    pub survey_confidence: f64,
}

impl Stratum {
    pub fn for_material(
        top_depth_m: f64,
        bottom_depth_m: f64,
        material: MaterialClass,
        survey_confidence: f64,
    ) -> Self {
        let (hardness, abrasiveness, permeability, gas_potential, roof_cohesion, ore_grade) =
            match material {
                MaterialClass::Unconsolidated => (0.12, 0.18, 0.82, 0.08, 0.18, 0.02),
                MaterialClass::Clay => (0.24, 0.12, 0.22, 0.12, 0.52, 0.03),
                MaterialClass::Sandstone => (0.48, 0.52, 0.56, 0.18, 0.68, 0.08),
                MaterialClass::Limestone => (0.58, 0.46, 0.68, 0.16, 0.72, 0.1),
                MaterialClass::Shale => (0.42, 0.38, 0.18, 0.62, 0.42, 0.06),
                MaterialClass::Granite => (0.92, 0.88, 0.04, 0.02, 0.94, 0.04),
                MaterialClass::OreBody => (0.72, 0.78, 0.2, 0.2, 0.76, 0.92),
                MaterialClass::FaultGouge => (0.2, 0.34, 0.9, 0.55, 0.12, 0.12),
            };
        Self {
            top_depth_m,
            bottom_depth_m,
            material,
            hardness,
            abrasiveness,
            permeability,
            gas_potential,
            roof_cohesion,
            ore_grade,
            survey_confidence,
        }
    }

    pub fn thickness_m(self) -> f64 {
        self.bottom_depth_m - self.top_depth_m
    }

    pub fn contains_depth(self, depth_m: f64) -> bool {
        depth_m >= self.top_depth_m && depth_m < self.bottom_depth_m
    }

    fn properties_are_valid(self) -> bool {
        [
            self.hardness,
            self.abrasiveness,
            self.permeability,
            self.gas_potential,
            self.roof_cohesion,
            self.ore_grade,
            self.survey_confidence,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeologySample {
    pub depth_m: f64,
    pub stratum_index: usize,
    pub material: MaterialClass,
    pub hardness: f64,
    pub abrasiveness: f64,
    pub permeability: f64,
    pub gas_potential: f64,
    pub roof_cohesion: f64,
    pub ore_grade: f64,
    pub survey_confidence: f64,
    pub distance_to_boundary_m: f64,
}

impl GeologySample {
    fn from_stratum(depth_m: f64, stratum_index: usize, stratum: Stratum) -> Self {
        Self {
            depth_m,
            stratum_index,
            material: stratum.material,
            hardness: stratum.hardness,
            abrasiveness: stratum.abrasiveness,
            permeability: stratum.permeability,
            gas_potential: stratum.gas_potential,
            roof_cohesion: stratum.roof_cohesion,
            ore_grade: stratum.ore_grade,
            survey_confidence: stratum.survey_confidence,
            distance_to_boundary_m: (stratum.bottom_depth_m - depth_m).max(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeologicalLookahead {
    pub start_depth_m: f64,
    pub horizon_m: f64,
    pub sampled_strata: usize,
    pub transition_count: usize,
    pub max_hardness: f64,
    pub max_permeability: f64,
    pub max_gas_potential: f64,
    pub minimum_roof_cohesion: f64,
    pub minimum_survey_confidence: f64,
    pub risk_score: f64,
    pub probe_required: bool,
}

impl GeologicalLookahead {
    pub const fn clear(depth_m: f64, horizon_m: f64) -> Self {
        Self {
            start_depth_m: depth_m,
            horizon_m,
            sampled_strata: 0,
            transition_count: 0,
            max_hardness: 0.0,
            max_permeability: 0.0,
            max_gas_potential: 0.0,
            minimum_roof_cohesion: 1.0,
            minimum_survey_confidence: 1.0,
            risk_score: 0.0,
            probe_required: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeologyError {
    EmptyProfile,
    NonFiniteDepth,
    NonPositiveThickness { index: usize },
    InvalidProperty { index: usize },
    NonContiguous { index: usize },
    SurfaceNotCovered,
    MaximumDepthNotCovered,
}

impl std::fmt::Display for GeologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProfile => {
                f.write_str("geotechnical profile must contain at least one stratum")
            }
            Self::NonFiniteDepth => f.write_str("stratum depths must be finite"),
            Self::NonPositiveThickness { index } => {
                write!(f, "stratum {index} must have positive thickness")
            }
            Self::InvalidProperty { index } => {
                write!(f, "stratum {index} properties must be finite and in [0, 1]")
            }
            Self::NonContiguous { index } => {
                write!(
                    f,
                    "stratum {index} is not contiguous with the preceding stratum"
                )
            }
            Self::SurfaceNotCovered => f.write_str("geotechnical profile must begin at depth 0 m"),
            Self::MaximumDepthNotCovered => write!(
                f,
                "geotechnical profile must cover at least {MAX_SIMULATED_DEPTH_M} m"
            ),
        }
    }
}

impl std::error::Error for GeologyError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeotechnicalProfile {
    strata: Vec<Stratum>,
}

impl GeotechnicalProfile {
    pub fn try_new(strata: Vec<Stratum>) -> Result<Self, GeologyError> {
        if strata.is_empty() {
            return Err(GeologyError::EmptyProfile);
        }
        if strata
            .iter()
            .any(|stratum| !stratum.top_depth_m.is_finite() || !stratum.bottom_depth_m.is_finite())
        {
            return Err(GeologyError::NonFiniteDepth);
        }
        if strata[0].top_depth_m.abs() > CONTIGUITY_EPSILON_M {
            return Err(GeologyError::SurfaceNotCovered);
        }
        for (index, stratum) in strata.iter().copied().enumerate() {
            if stratum.thickness_m() <= 0.0 {
                return Err(GeologyError::NonPositiveThickness { index });
            }
            if !stratum.properties_are_valid() {
                return Err(GeologyError::InvalidProperty { index });
            }
            if index > 0 {
                let preceding = strata[index - 1];
                if (preceding.bottom_depth_m - stratum.top_depth_m).abs() > CONTIGUITY_EPSILON_M {
                    return Err(GeologyError::NonContiguous { index });
                }
            }
        }
        if strata[strata.len() - 1].bottom_depth_m + CONTIGUITY_EPSILON_M < MAX_SIMULATED_DEPTH_M {
            return Err(GeologyError::MaximumDepthNotCovered);
        }
        Ok(Self { strata })
    }

    pub fn reference() -> Self {
        Self::try_new(vec![
            Stratum::for_material(0.0, 8.0, MaterialClass::Unconsolidated, 0.98),
            Stratum::for_material(8.0, 28.0, MaterialClass::Clay, 0.94),
            Stratum::for_material(28.0, 62.0, MaterialClass::Sandstone, 0.9),
            Stratum::for_material(62.0, 86.0, MaterialClass::Limestone, 0.82),
            Stratum::for_material(86.0, 105.0, MaterialClass::FaultGouge, 0.58),
            Stratum::for_material(105.0, 142.0, MaterialClass::Shale, 0.76),
            Stratum::for_material(142.0, 168.0, MaterialClass::OreBody, 0.72),
            Stratum::for_material(168.0, 220.0, MaterialClass::Granite, 0.88),
        ])
        .unwrap_or_else(|_| Self {
            strata: vec![Stratum::for_material(
                0.0,
                MAX_SIMULATED_DEPTH_M,
                MaterialClass::Sandstone,
                1.0,
            )],
        })
    }

    pub fn homogeneous(material: MaterialClass) -> Self {
        Self {
            strata: vec![Stratum::for_material(
                0.0,
                MAX_SIMULATED_DEPTH_M,
                material,
                1.0,
            )],
        }
    }

    pub fn lookahead(&self, depth_m: f64, horizon_m: f64) -> GeologicalLookahead {
        let start_depth_m = if depth_m.is_finite() {
            depth_m.clamp(0.0, MAX_SIMULATED_DEPTH_M)
        } else {
            0.0
        };
        let horizon_m = if horizon_m.is_finite() && horizon_m > 0.0 {
            horizon_m.min(MAX_SIMULATED_DEPTH_M)
        } else {
            1.0
        };
        let end_depth_m = (start_depth_m + horizon_m).min(MAX_SIMULATED_DEPTH_M);
        let mut result = GeologicalLookahead::clear(start_depth_m, horizon_m);
        let mut previous_index = None;
        for (index, stratum) in self.strata.iter().copied().enumerate() {
            let overlaps =
                stratum.bottom_depth_m > start_depth_m && stratum.top_depth_m <= end_depth_m;
            if !overlaps {
                continue;
            }
            result.sampled_strata += 1;
            if previous_index.is_some_and(|previous| previous != index) {
                result.transition_count += 1;
            }
            previous_index = Some(index);
            result.max_hardness = result.max_hardness.max(stratum.hardness);
            result.max_permeability = result.max_permeability.max(stratum.permeability);
            result.max_gas_potential = result.max_gas_potential.max(stratum.gas_potential);
            result.minimum_roof_cohesion = result.minimum_roof_cohesion.min(stratum.roof_cohesion);
            result.minimum_survey_confidence = result
                .minimum_survey_confidence
                .min(stratum.survey_confidence);
        }
        result.risk_score = (result.max_permeability * 0.26
            + result.max_gas_potential * 0.26
            + (1.0 - result.minimum_roof_cohesion) * 0.26
            + (1.0 - result.minimum_survey_confidence) * 0.16
            + (result.transition_count.min(2) as f64) * 0.03)
            .clamp(0.0, 1.0);
        result.probe_required = result.minimum_survey_confidence < 0.68
            || result.risk_score >= 0.58
            || (result.transition_count > 0 && result.risk_score >= 0.45);
        result
    }

    pub fn strata(&self) -> &[Stratum] {
        &self.strata
    }

    pub fn sample(&self, depth_m: f64) -> GeologySample {
        let depth_m = if depth_m.is_finite() {
            depth_m.clamp(0.0, MAX_SIMULATED_DEPTH_M)
        } else {
            0.0
        };
        let index = self
            .strata
            .iter()
            .position(|stratum| stratum.contains_depth(depth_m))
            .unwrap_or(self.strata.len() - 1);
        GeologySample::from_stratum(depth_m, index, self.strata[index])
    }
}

impl Default for GeotechnicalProfile {
    fn default() -> Self {
        Self::reference()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_profile_is_contiguous_and_covers_simulator_depth() {
        let profile = GeotechnicalProfile::reference();
        assert!(profile.strata().len() >= 6);
        assert_eq!(profile.sample(0.0).material, MaterialClass::Unconsolidated);
        assert_eq!(profile.sample(180.0).material, MaterialClass::Granite);
    }

    #[test]
    fn profile_rejects_gaps_and_invalid_properties() {
        let gap = GeotechnicalProfile::try_new(vec![
            Stratum::for_material(0.0, 20.0, MaterialClass::Clay, 1.0),
            Stratum::for_material(21.0, 220.0, MaterialClass::Granite, 1.0),
        ]);
        assert!(matches!(gap, Err(GeologyError::NonContiguous { index: 1 })));

        let mut invalid = Stratum::for_material(0.0, 220.0, MaterialClass::Clay, 1.0);
        invalid.permeability = 1.2;
        assert!(matches!(
            GeotechnicalProfile::try_new(vec![invalid]),
            Err(GeologyError::InvalidProperty { index: 0 })
        ));
    }

    #[test]
    fn homogeneous_material_presets_have_distinct_operational_properties() {
        let granite = GeotechnicalProfile::homogeneous(MaterialClass::Granite).sample(50.0);
        let clay = GeotechnicalProfile::homogeneous(MaterialClass::Clay).sample(50.0);
        assert!(granite.hardness > clay.hardness);
        assert!(granite.abrasiveness > clay.abrasiveness);
        assert!(clay.permeability > granite.permeability);
    }

    #[test]
    fn lookahead_detects_uncertain_fault_transition_before_entry() {
        let profile = GeotechnicalProfile::reference();
        let lookahead = profile.lookahead(82.0, 8.0);
        assert!(lookahead.transition_count >= 1);
        assert!(lookahead.max_permeability > 0.8);
        assert!(lookahead.minimum_survey_confidence < 0.7);
        assert!(lookahead.probe_required);
    }
}
