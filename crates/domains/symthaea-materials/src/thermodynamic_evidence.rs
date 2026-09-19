// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Normalized thermodynamic evidence for materials discovery.
//!
//! External databases are valuable corroboration, but a database record is not
//! equivalent to a locally reproduced first-principles calculation. This module
//! preserves that distinction while normalizing common thermodynamic quantities
//! into eV/atom for comparison and downstream evidence handling.

use crate::evidence::{
    MaterialsEvidenceError, MaterialsEvidenceKind, MaterialsEvidenceRecord, MaterialsEvidenceStage,
};
use serde::{Deserialize, Serialize};
use symthaea_epistemic_types::EpistemicCoordinate;

const NUMERICAL_ZERO_TOLERANCE_EV_ATOM: f64 = 1.0e-6;

/// Supported external thermodynamic materials databases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermodynamicDatabase {
    /// Materials Project.
    MaterialsProject,
    /// Open Quantum Materials Database.
    Oqmd,
    /// NOMAD Repository / Archive.
    Nomad,
    /// Another explicitly identified database.
    Other,
}

/// Provenance class for a normalized thermodynamic observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermodynamicEvidenceOrigin {
    /// Evidence retrieved from an external database.
    ExternalDatabase {
        /// Database that supplied the record.
        database: ThermodynamicDatabase,
        /// Database release, snapshot, API revision, or retrieval label when available.
        dataset_version: Option<String>,
    },
    /// Evidence produced by a locally controlled first-principles calculation.
    LocalDftCalculation {
        /// Electronic-structure code and version.
        code: String,
        /// Exchange-correlation functional or equivalent method label.
        functional: String,
        /// Digest of the relaxed/input structure used for the calculation.
        structure_digest: String,
        /// Digest binding the complete calculation input deck.
        input_digest: String,
        /// Digest binding the raw or canonicalized calculation outputs.
        output_digest: String,
    },
}

impl ThermodynamicEvidenceOrigin {
    /// Whether the origin is an external database observation.
    pub fn is_external_database(&self) -> bool {
        matches!(self, Self::ExternalDatabase { .. })
    }

    /// Whether the origin is a reproducibility-bound local DFT calculation.
    pub fn is_local_dft(&self) -> bool {
        matches!(self, Self::LocalDftCalculation { .. })
    }
}

/// Thermodynamic quantities normalized to eV/atom.
///
/// Not every provider exposes every field. Missing values remain `None`; they
/// must never be silently filled from heuristic estimates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedThermodynamicEvidence {
    /// Stable source identifier: material ID, calculation ID, archive ID, etc.
    pub source_id: String,
    /// Canonical or provider formula associated with this evidence record.
    pub formula: String,
    /// Evidence provenance class.
    pub origin: ThermodynamicEvidenceOrigin,
    /// Formation energy in eV/atom when supplied by the source/calculation.
    pub formation_energy_ev_atom: Option<f64>,
    /// Energy above the convex hull in eV/atom when supplied by the source/calculation.
    pub energy_above_hull_ev_atom: Option<f64>,
    /// Decomposition energy in eV/atom when separately supplied.
    pub decomposition_energy_ev_atom: Option<f64>,
    /// Temperature represented by the thermodynamic result, if explicitly defined.
    pub temperature_k: Option<f64>,
    /// Human-readable method/provenance note.
    pub method: String,
    /// Canonical Symthaea epistemic coordinate for this evidence item.
    pub epistemic: EpistemicCoordinate,
    /// Optional digest of a captured provider response or evidence bundle.
    pub artifact_digest: Option<String>,
}

impl NormalizedThermodynamicEvidence {
    /// Validate unit normalization, provenance completeness, and finite numeric values.
    pub fn validate(&self) -> Result<(), ThermodynamicEvidenceError> {
        if self.source_id.trim().is_empty() {
            return Err(ThermodynamicEvidenceError::EmptySourceId);
        }
        if self.formula.trim().is_empty() {
            return Err(ThermodynamicEvidenceError::EmptyFormula);
        }
        if self.method.trim().is_empty() {
            return Err(ThermodynamicEvidenceError::EmptyMethod);
        }

        validate_optional_finite("formation_energy_ev_atom", self.formation_energy_ev_atom)?;
        validate_optional_finite(
            "energy_above_hull_ev_atom",
            self.energy_above_hull_ev_atom,
        )?;
        validate_optional_finite(
            "decomposition_energy_ev_atom",
            self.decomposition_energy_ev_atom,
        )?;
        validate_optional_finite("temperature_k", self.temperature_k)?;

        if let Some(hull) = self.energy_above_hull_ev_atom {
            if hull < -NUMERICAL_ZERO_TOLERANCE_EV_ATOM {
                return Err(ThermodynamicEvidenceError::NegativeHullDistance { value: hull });
            }
        }
        if let Some(t) = self.temperature_k {
            if t < 0.0 {
                return Err(ThermodynamicEvidenceError::NegativeTemperature { value: t });
            }
        }

        match &self.origin {
            ThermodynamicEvidenceOrigin::ExternalDatabase { .. } => {}
            ThermodynamicEvidenceOrigin::LocalDftCalculation {
                code,
                functional,
                structure_digest,
                input_digest,
                output_digest,
            } => {
                for (field, value) in [
                    ("code", code),
                    ("functional", functional),
                    ("structure_digest", structure_digest),
                    ("input_digest", input_digest),
                    ("output_digest", output_digest),
                ] {
                    if value.trim().is_empty() {
                        return Err(ThermodynamicEvidenceError::MissingLocalDftBinding { field });
                    }
                }
            }
        }

        if self.formation_energy_ev_atom.is_none()
            && self.energy_above_hull_ev_atom.is_none()
            && self.decomposition_energy_ev_atom.is_none()
        {
            return Err(ThermodynamicEvidenceError::NoThermodynamicQuantity);
        }

        Ok(())
    }

    /// Convert external-database evidence into a materials evidence record.
    ///
    /// The strongest permitted stage is intentionally `DatabaseCorroborated`.
    /// Even if the database itself contains DFT-derived values, retrieving those
    /// values does not demonstrate that Symthaea reproduced the calculation.
    pub fn as_database_evidence_record(
        &self,
    ) -> Result<MaterialsEvidenceRecord, ThermodynamicEvidenceError> {
        self.validate()?;
        if !self.origin.is_external_database() {
            return Err(ThermodynamicEvidenceError::ExpectedExternalDatabase);
        }
        Ok(MaterialsEvidenceRecord {
            stage: MaterialsEvidenceStage::DatabaseCorroborated,
            kind: MaterialsEvidenceKind::ExternalDatabase,
            source_id: self.source_id.clone(),
            method: self.method.clone(),
            epistemic: self.epistemic,
            artifact_digest: self.artifact_digest.clone(),
        })
    }

    /// Convert a bound local DFT calculation into formation-stability evidence.
    ///
    /// A negative formation energy is required for this specific stage. Convex-hull
    /// qualification remains a separate later stage and is never inferred here.
    pub fn as_dft_formation_evidence_record(
        &self,
    ) -> Result<MaterialsEvidenceRecord, ThermodynamicEvidenceError> {
        self.validate()?;
        if !self.origin.is_local_dft() {
            return Err(ThermodynamicEvidenceError::ExpectedLocalDft);
        }
        let formation = self
            .formation_energy_ev_atom
            .ok_or(ThermodynamicEvidenceError::MissingFormationEnergy)?;
        if formation >= 0.0 {
            return Err(ThermodynamicEvidenceError::FormationEnergyNotNegative {
                value: formation,
            });
        }
        Ok(MaterialsEvidenceRecord {
            stage: MaterialsEvidenceStage::DftFormationStable,
            kind: MaterialsEvidenceKind::DftCalculation,
            source_id: self.source_id.clone(),
            method: self.method.clone(),
            epistemic: self.epistemic,
            artifact_digest: self.artifact_digest.clone(),
        })
    }

    /// Return the hull distance after clamping tiny negative numerical noise to zero.
    pub fn normalized_hull_distance_ev_atom(&self) -> Option<f64> {
        self.energy_above_hull_ev_atom.map(|value| {
            if value < 0.0 && value >= -NUMERICAL_ZERO_TOLERANCE_EV_ATOM {
                0.0
            } else {
                value
            }
        })
    }
}

fn validate_optional_finite(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), ThermodynamicEvidenceError> {
    if let Some(v) = value {
        if !v.is_finite() {
            return Err(ThermodynamicEvidenceError::NonFiniteValue { field, value: v });
        }
    }
    Ok(())
}

/// Validation/conversion failures for normalized thermodynamic evidence.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermodynamicEvidenceError {
    /// Source identifiers must not be empty.
    EmptySourceId,
    /// Formula identifiers must not be empty.
    EmptyFormula,
    /// Method descriptions must not be empty.
    EmptyMethod,
    /// A numeric field contained NaN or infinity.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Hull distance was physically invalid beyond numerical tolerance.
    NegativeHullDistance {
        /// Invalid hull distance in eV/atom.
        value: f64,
    },
    /// Temperature must be non-negative.
    NegativeTemperature {
        /// Invalid temperature in kelvin.
        value: f64,
    },
    /// A required local-DFT reproducibility binding was absent.
    MissingLocalDftBinding {
        /// Missing field name.
        field: &'static str,
    },
    /// No actual thermodynamic quantity was supplied.
    NoThermodynamicQuantity,
    /// Conversion required an external database origin.
    ExpectedExternalDatabase,
    /// Conversion required a locally bound DFT origin.
    ExpectedLocalDft,
    /// Formation energy is required for DFT formation-stability evidence.
    MissingFormationEnergy,
    /// Formation energy did not support exothermic formation.
    FormationEnergyNotNegative {
        /// Formation energy in eV/atom.
        value: f64,
    },
    /// The resulting generic materials evidence record failed validation.
    MaterialsEvidence(MaterialsEvidenceError),
}

impl From<MaterialsEvidenceError> for ThermodynamicEvidenceError {
    fn from(value: MaterialsEvidenceError) -> Self {
        Self::MaterialsEvidence(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{
        EmpiricalLevel, EpistemicContext, MaterialityLevel, NormativeLevel,
    };

    fn coordinate() -> EpistemicCoordinate {
        EpistemicCoordinate {
            empirical: EmpiricalLevel::E0Null,
            normative: NormativeLevel::N0Personal,
            materiality: MaterialityLevel::M1Temporal,
            context: EpistemicContext::Scientific,
        }
    }

    fn external() -> NormalizedThermodynamicEvidence {
        NormalizedThermodynamicEvidence {
            source_id: "mp-test".to_string(),
            formula: "TiZrNbTa".to_string(),
            origin: ThermodynamicEvidenceOrigin::ExternalDatabase {
                database: ThermodynamicDatabase::MaterialsProject,
                dataset_version: Some("test-snapshot".to_string()),
            },
            formation_energy_ev_atom: Some(-0.25),
            energy_above_hull_ev_atom: Some(0.0),
            decomposition_energy_ev_atom: None,
            temperature_k: Some(0.0),
            method: "retrieved thermodynamic database record".to_string(),
            epistemic: coordinate(),
            artifact_digest: Some("sha256:test".to_string()),
        }
    }

    fn local_dft() -> NormalizedThermodynamicEvidence {
        NormalizedThermodynamicEvidence {
            source_id: "calc-001".to_string(),
            formula: "TiZrNbTa".to_string(),
            origin: ThermodynamicEvidenceOrigin::LocalDftCalculation {
                code: "VASP 6.x".to_string(),
                functional: "PBE".to_string(),
                structure_digest: "sha256:structure".to_string(),
                input_digest: "sha256:inputs".to_string(),
                output_digest: "sha256:outputs".to_string(),
            },
            formation_energy_ev_atom: Some(-0.18),
            energy_above_hull_ev_atom: None,
            decomposition_energy_ev_atom: None,
            temperature_k: Some(0.0),
            method: "bound local DFT formation-energy calculation".to_string(),
            epistemic: coordinate(),
            artifact_digest: Some("sha256:bundle".to_string()),
        }
    }

    #[test]
    fn external_database_never_auto_promotes_to_dft_stage() {
        let record = external().as_database_evidence_record().unwrap();
        assert_eq!(record.stage, MaterialsEvidenceStage::DatabaseCorroborated);
        assert_eq!(record.kind, MaterialsEvidenceKind::ExternalDatabase);
    }

    #[test]
    fn external_database_cannot_be_recast_as_local_dft() {
        assert!(matches!(
            external().as_dft_formation_evidence_record(),
            Err(ThermodynamicEvidenceError::ExpectedLocalDft)
        ));
    }

    #[test]
    fn local_dft_cannot_be_recast_as_database_corroboration() {
        assert!(matches!(
            local_dft().as_database_evidence_record(),
            Err(ThermodynamicEvidenceError::ExpectedExternalDatabase)
        ));
    }

    #[test]
    fn reproducibly_bound_negative_formation_energy_can_reach_dft_stage() {
        let record = local_dft().as_dft_formation_evidence_record().unwrap();
        assert_eq!(record.stage, MaterialsEvidenceStage::DftFormationStable);
        assert_eq!(record.kind, MaterialsEvidenceKind::DftCalculation);
    }

    #[test]
    fn positive_formation_energy_does_not_qualify_dft_formation_stage() {
        let mut evidence = local_dft();
        evidence.formation_energy_ev_atom = Some(0.05);
        assert!(matches!(
            evidence.as_dft_formation_evidence_record(),
            Err(ThermodynamicEvidenceError::FormationEnergyNotNegative { .. })
        ));
    }

    #[test]
    fn tiny_negative_hull_noise_is_normalized_to_zero() {
        let mut evidence = external();
        evidence.energy_above_hull_ev_atom = Some(-5.0e-7);
        evidence.validate().unwrap();
        assert_eq!(evidence.normalized_hull_distance_ev_atom(), Some(0.0));
    }

    #[test]
    fn materially_negative_hull_distance_is_rejected() {
        let mut evidence = external();
        evidence.energy_above_hull_ev_atom = Some(-1.0e-3);
        assert!(matches!(
            evidence.validate(),
            Err(ThermodynamicEvidenceError::NegativeHullDistance { .. })
        ));
    }

    #[test]
    fn local_dft_requires_reproducibility_bindings() {
        let mut evidence = local_dft();
        if let ThermodynamicEvidenceOrigin::LocalDftCalculation { input_digest, .. } =
            &mut evidence.origin
        {
            input_digest.clear();
        }
        assert!(matches!(
            evidence.validate(),
            Err(ThermodynamicEvidenceError::MissingLocalDftBinding {
                field: "input_digest"
            })
        ));
    }

    #[test]
    fn evidence_requires_at_least_one_thermodynamic_quantity() {
        let mut evidence = external();
        evidence.formation_energy_ev_atom = None;
        evidence.energy_above_hull_ev_atom = None;
        evidence.decomposition_energy_ev_atom = None;
        assert!(matches!(
            evidence.validate(),
            Err(ThermodynamicEvidenceError::NoThermodynamicQuantity)
        ));
    }
}