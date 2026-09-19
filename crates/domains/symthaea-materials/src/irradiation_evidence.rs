// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Composition- and spectrum-resolved irradiation evidence.
//!
//! This module intentionally stops at a displacement exposure index.  It does
//! not infer lifetime, embrittlement, swelling, or "self-healing" from dpa.
//! Such property claims require separate evidence models and validation.
//!
//! The calculation assumes each supplied displacement cross-section bin is a
//! spectrum-compatible displacement-damage cross section in barns for the named
//! model (for example NRT-dpa or arc-dpa).  The neutron flux for each spectrum
//! bin is the bin-integrated flux in n/cm²/s, so
//!
//! `dpa_rate = Σ_i x_i Σ_b phi_b * sigma_d(i,b) * 1e-24`.
//!
//! ## References / conventions
//!
//! - ASTM E693 describes spectrum-sensitive dpa exposure calculations and
//!   explicitly cautions that dpa does not map directly to one material-property
//!   change.
//! - arc-dpa corrects primary-defect production for athermal recombination and
//!   must remain distinguishable from NRT-dpa rather than being mixed silently.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Exact integer denominator for composition fractions.
pub const IRRADIATION_COMPOSITION_PPM_TOTAL: u32 = 1_000_000;
const BARN_TO_CM2: f64 = 1.0e-24;
const FRACTION_TOLERANCE_PPM: u32 = 0;

/// Named displacement-exposure model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DamageMetricModel {
    /// Norgett-Robinson-Torrens displacement-per-atom convention.
    NrtDpa,
    /// Athermal-recombination-corrected displacement-per-atom convention.
    ArcDpa,
    /// Another explicitly named convention. The caller remains responsible for
    /// proving that all curves in one calculation use the same convention.
    Other(String),
}

/// Exact composition entry for an irradiation subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrradiationCompositionComponent {
    /// Atomic number.
    pub atomic_number: u16,
    /// Atomic fraction in integer ppm of the whole subject composition.
    pub fraction_ppm: u32,
}

/// One bin of an incident neutron spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeutronSpectrumBin {
    /// Inclusive lower energy bound in eV.
    pub energy_lower_ev: f64,
    /// Exclusive upper energy bound in eV.
    pub energy_upper_ev: f64,
    /// Bin-integrated neutron flux in n/cm²/s.
    pub flux_n_cm2_s: f64,
}

/// One energy-bin value of a displacement cross-section curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisplacementCrossSectionBin {
    /// Lower energy bound in eV; must match the corresponding spectrum bin.
    pub energy_lower_ev: f64,
    /// Upper energy bound in eV; must match the corresponding spectrum bin.
    pub energy_upper_ev: f64,
    /// Displacement cross section in barns for the declared damage model.
    pub displacement_cross_section_barn: f64,
}

/// Source-bound displacement cross sections for one constituent element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementDisplacementCurve {
    /// Element atomic number.
    pub atomic_number: u16,
    /// Damage metric convention used to generate these cross sections.
    pub model: DamageMetricModel,
    /// Threshold displacement energy used by the source calculation, in eV.
    pub threshold_displacement_energy_ev: f64,
    /// Human/machine-readable source identifier (dataset, calculation, paper, etc.).
    pub source_id: String,
    /// SHA-256 digest of the exact source artifact used for this curve.
    pub artifact_sha256: String,
    /// Energy-binned displacement cross section.
    pub bins: Vec<DisplacementCrossSectionBin>,
}

/// Complete irradiation calculation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrradiationEvidenceInput {
    /// Stable subject/candidate identity.
    pub subject_id: String,
    /// Exact composition. Fractions must sum to one million ppm.
    pub composition: Vec<IrradiationCompositionComponent>,
    /// Material temperature during exposure in kelvin.
    pub temperature_k: f64,
    /// Exposure duration in seconds.
    pub exposure_s: f64,
    /// Identifier for the neutron-spectrum source.
    pub spectrum_source_id: String,
    /// SHA-256 digest of the exact spectrum artifact.
    pub spectrum_artifact_sha256: String,
    /// Incident neutron spectrum.
    pub spectrum: Vec<NeutronSpectrumBin>,
    /// One complete, spectrum-aligned curve per constituent element.
    pub displacement_curves: Vec<ElementDisplacementCurve>,
}

/// Per-element contribution retained so composition effects stay inspectable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementDpaContribution {
    /// Atomic number.
    pub atomic_number: u16,
    /// Subject atomic fraction in ppm.
    pub fraction_ppm: u32,
    /// Composition-weighted dpa rate contribution in s^-1.
    pub dpa_rate_per_s: f64,
    /// Composition-weighted dpa over the requested exposure.
    pub total_dpa: f64,
    /// Cross-section source identifier.
    pub source_id: String,
    /// Exact source-artifact digest.
    pub artifact_sha256: String,
}

/// Evidence-bearing displacement exposure result.
///
/// Deliberately contains no material lifetime or self-healing field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionDpaEvidence {
    /// Subject identity copied from the validated request.
    pub subject_id: String,
    /// Damage convention shared by all element curves.
    pub model: DamageMetricModel,
    /// Temperature associated with the exposure.
    pub temperature_k: f64,
    /// Exposure duration.
    pub exposure_s: f64,
    /// Composition-resolved dpa rate in s^-1.
    pub dpa_rate_per_s: f64,
    /// Total dpa exposure = rate × duration.
    pub total_dpa: f64,
    /// Per-element contributions.
    pub elemental_contributions: Vec<ElementDpaContribution>,
    /// Spectrum source identifier.
    pub spectrum_source_id: String,
    /// Spectrum artifact digest.
    pub spectrum_artifact_sha256: String,
}

/// Calculate composition-resolved dpa evidence from externally supplied curves.
pub fn calculate_composition_dpa_evidence(
    input: &IrradiationEvidenceInput,
) -> Result<CompositionDpaEvidence, IrradiationEvidenceError> {
    validate_input(input)?;

    let model = input.displacement_curves[0].model.clone();
    let curves: HashMap<u16, &ElementDisplacementCurve> = input
        .displacement_curves
        .iter()
        .map(|curve| (curve.atomic_number, curve))
        .collect();

    let mut elemental_contributions = Vec::with_capacity(input.composition.len());
    let mut total_rate = 0.0;

    for component in &input.composition {
        let curve = curves
            .get(&component.atomic_number)
            .ok_or(IrradiationEvidenceError::MissingElementCurve(
                component.atomic_number,
            ))?;
        let atomic_fraction = component.fraction_ppm as f64
            / IRRADIATION_COMPOSITION_PPM_TOTAL as f64;
        let unweighted_rate: f64 = input
            .spectrum
            .iter()
            .zip(curve.bins.iter())
            .map(|(spectrum_bin, curve_bin)| {
                spectrum_bin.flux_n_cm2_s
                    * curve_bin.displacement_cross_section_barn
                    * BARN_TO_CM2
            })
            .sum();
        let rate = atomic_fraction * unweighted_rate;
        let total_dpa = rate * input.exposure_s;
        total_rate += rate;
        elemental_contributions.push(ElementDpaContribution {
            atomic_number: component.atomic_number,
            fraction_ppm: component.fraction_ppm,
            dpa_rate_per_s: rate,
            total_dpa,
            source_id: curve.source_id.clone(),
            artifact_sha256: curve.artifact_sha256.clone(),
        });
    }

    Ok(CompositionDpaEvidence {
        subject_id: input.subject_id.clone(),
        model,
        temperature_k: input.temperature_k,
        exposure_s: input.exposure_s,
        dpa_rate_per_s: total_rate,
        total_dpa: total_rate * input.exposure_s,
        elemental_contributions,
        spectrum_source_id: input.spectrum_source_id.clone(),
        spectrum_artifact_sha256: input.spectrum_artifact_sha256.clone(),
    })
}

fn validate_input(input: &IrradiationEvidenceInput) -> Result<(), IrradiationEvidenceError> {
    if input.subject_id.trim().is_empty() {
        return Err(IrradiationEvidenceError::EmptySubjectId);
    }
    validate_positive_finite("temperature_k", input.temperature_k)?;
    validate_positive_finite("exposure_s", input.exposure_s)?;
    validate_source_binding(
        "spectrum",
        &input.spectrum_source_id,
        &input.spectrum_artifact_sha256,
    )?;

    if input.composition.is_empty() {
        return Err(IrradiationEvidenceError::EmptyComposition);
    }
    let mut composition_elements = HashSet::new();
    let mut fraction_sum: u32 = 0;
    for component in &input.composition {
        if component.fraction_ppm == 0 {
            return Err(IrradiationEvidenceError::ZeroCompositionFraction(
                component.atomic_number,
            ));
        }
        if !composition_elements.insert(component.atomic_number) {
            return Err(IrradiationEvidenceError::DuplicateCompositionElement(
                component.atomic_number,
            ));
        }
        fraction_sum = fraction_sum
            .checked_add(component.fraction_ppm)
            .ok_or(IrradiationEvidenceError::CompositionFractionOverflow)?;
    }
    if fraction_sum.abs_diff(IRRADIATION_COMPOSITION_PPM_TOTAL) > FRACTION_TOLERANCE_PPM {
        return Err(IrradiationEvidenceError::CompositionDoesNotSumToOne {
            sum_ppm: fraction_sum,
        });
    }

    if input.spectrum.is_empty() {
        return Err(IrradiationEvidenceError::EmptySpectrum);
    }
    let mut previous_upper = None;
    for (index, bin) in input.spectrum.iter().enumerate() {
        validate_energy_bin(index, bin.energy_lower_ev, bin.energy_upper_ev)?;
        validate_nonnegative_finite("flux_n_cm2_s", bin.flux_n_cm2_s)?;
        if let Some(previous) = previous_upper {
            if !close(previous, bin.energy_lower_ev) {
                return Err(IrradiationEvidenceError::NonContiguousSpectrum { index });
            }
        }
        previous_upper = Some(bin.energy_upper_ev);
    }

    if input.displacement_curves.len() != input.composition.len() {
        return Err(IrradiationEvidenceError::CurveCountMismatch {
            expected: input.composition.len(),
            actual: input.displacement_curves.len(),
        });
    }

    let mut curve_elements = HashSet::new();
    let mut model: Option<&DamageMetricModel> = None;
    for curve in &input.displacement_curves {
        if !curve_elements.insert(curve.atomic_number) {
            return Err(IrradiationEvidenceError::DuplicateElementCurve(
                curve.atomic_number,
            ));
        }
        if !composition_elements.contains(&curve.atomic_number) {
            return Err(IrradiationEvidenceError::CurveElementOutsideComposition(
                curve.atomic_number,
            ));
        }
        validate_positive_finite(
            "threshold_displacement_energy_ev",
            curve.threshold_displacement_energy_ev,
        )?;
        validate_source_binding("curve", &curve.source_id, &curve.artifact_sha256)?;
        match model {
            Some(expected) if expected != &curve.model => {
                return Err(IrradiationEvidenceError::MixedDamageModels)
            }
            None => model = Some(&curve.model),
            _ => {}
        }
        if curve.bins.len() != input.spectrum.len() {
            return Err(IrradiationEvidenceError::CurveBinCountMismatch {
                atomic_number: curve.atomic_number,
                expected: input.spectrum.len(),
                actual: curve.bins.len(),
            });
        }
        for (index, (spectrum_bin, curve_bin)) in
            input.spectrum.iter().zip(curve.bins.iter()).enumerate()
        {
            validate_energy_bin(index, curve_bin.energy_lower_ev, curve_bin.energy_upper_ev)?;
            validate_nonnegative_finite(
                "displacement_cross_section_barn",
                curve_bin.displacement_cross_section_barn,
            )?;
            if !close(spectrum_bin.energy_lower_ev, curve_bin.energy_lower_ev)
                || !close(spectrum_bin.energy_upper_ev, curve_bin.energy_upper_ev)
            {
                return Err(IrradiationEvidenceError::EnergyGridMismatch {
                    atomic_number: curve.atomic_number,
                    index,
                });
            }
        }
    }

    for atomic_number in composition_elements {
        if !curve_elements.contains(&atomic_number) {
            return Err(IrradiationEvidenceError::MissingElementCurve(atomic_number));
        }
    }
    Ok(())
}

fn validate_energy_bin(index: usize, lower: f64, upper: f64) -> Result<(), IrradiationEvidenceError> {
    if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || upper <= lower {
        return Err(IrradiationEvidenceError::InvalidEnergyBin {
            index,
            lower_ev: lower,
            upper_ev: upper,
        });
    }
    Ok(())
}

fn validate_positive_finite(field: &'static str, value: f64) -> Result<(), IrradiationEvidenceError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(IrradiationEvidenceError::InvalidPositiveValue { field, value });
    }
    Ok(())
}

fn validate_nonnegative_finite(
    field: &'static str,
    value: f64,
) -> Result<(), IrradiationEvidenceError> {
    if !value.is_finite() || value < 0.0 {
        return Err(IrradiationEvidenceError::InvalidNonnegativeValue { field, value });
    }
    Ok(())
}

fn validate_source_binding(
    kind: &'static str,
    source_id: &str,
    artifact_sha256: &str,
) -> Result<(), IrradiationEvidenceError> {
    if source_id.trim().is_empty() {
        return Err(IrradiationEvidenceError::EmptySourceId { kind });
    }
    if artifact_sha256.len() != 64 || !artifact_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(IrradiationEvidenceError::InvalidSha256 { kind });
    }
    Ok(())
}

fn close(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 1.0e-12 * scale
}

/// Irradiation-evidence validation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum IrradiationEvidenceError {
    /// Subject identity was empty.
    EmptySubjectId,
    /// Composition was empty.
    EmptyComposition,
    /// One composition element had zero fraction.
    ZeroCompositionFraction(u16),
    /// Composition repeated an element.
    DuplicateCompositionElement(u16),
    /// Integer fraction accumulation overflowed.
    CompositionFractionOverflow,
    /// Exact composition did not sum to one million ppm.
    CompositionDoesNotSumToOne {
        /// Actual sum.
        sum_ppm: u32,
    },
    /// Spectrum was empty.
    EmptySpectrum,
    /// An energy bin was malformed.
    InvalidEnergyBin {
        /// Bin index.
        index: usize,
        /// Lower energy bound.
        lower_ev: f64,
        /// Upper energy bound.
        upper_ev: f64,
    },
    /// Spectrum bins were not contiguous.
    NonContiguousSpectrum {
        /// First non-contiguous bin index.
        index: usize,
    },
    /// Number of element curves differed from the composition cardinality.
    CurveCountMismatch {
        /// Expected curve count.
        expected: usize,
        /// Actual curve count.
        actual: usize,
    },
    /// Same element curve supplied more than once.
    DuplicateElementCurve(u16),
    /// Curve referred to an element not present in the subject.
    CurveElementOutsideComposition(u16),
    /// Subject contained an element for which no curve was supplied.
    MissingElementCurve(u16),
    /// Curves mixed incompatible dpa conventions.
    MixedDamageModels,
    /// Curve energy grid differed from the bound neutron-spectrum grid.
    EnergyGridMismatch {
        /// Element whose curve mismatched.
        atomic_number: u16,
        /// Mismatching bin index.
        index: usize,
    },
    /// Curve and spectrum had different numbers of energy bins.
    CurveBinCountMismatch {
        /// Element atomic number.
        atomic_number: u16,
        /// Expected number of bins.
        expected: usize,
        /// Actual number of bins.
        actual: usize,
    },
    /// Required positive finite scalar was invalid.
    InvalidPositiveValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Required nonnegative finite scalar was invalid.
    InvalidNonnegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Source identifier was empty.
    EmptySourceId {
        /// Source kind (`spectrum` or `curve`).
        kind: &'static str,
    },
    /// Artifact digest was not a 64-digit hexadecimal SHA-256 string.
    InvalidSha256 {
        /// Source kind (`spectrum` or `curve`).
        kind: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn spectrum() -> Vec<NeutronSpectrumBin> {
        vec![
            NeutronSpectrumBin {
                energy_lower_ev: 1.0e5,
                energy_upper_ev: 1.0e6,
                flux_n_cm2_s: 2.0e14,
            },
            NeutronSpectrumBin {
                energy_lower_ev: 1.0e6,
                energy_upper_ev: 15.0e6,
                flux_n_cm2_s: 1.0e14,
            },
        ]
    }

    fn curve(atomic_number: u16, sigma_a: f64, sigma_b: f64) -> ElementDisplacementCurve {
        ElementDisplacementCurve {
            atomic_number,
            model: DamageMetricModel::ArcDpa,
            threshold_displacement_energy_ev: 40.0,
            source_id: format!("fixture-Z{atomic_number}"),
            artifact_sha256: if atomic_number == 22 {
                DIGEST_B.to_string()
            } else {
                DIGEST_C.to_string()
            },
            bins: vec![
                DisplacementCrossSectionBin {
                    energy_lower_ev: 1.0e5,
                    energy_upper_ev: 1.0e6,
                    displacement_cross_section_barn: sigma_a,
                },
                DisplacementCrossSectionBin {
                    energy_lower_ev: 1.0e6,
                    energy_upper_ev: 15.0e6,
                    displacement_cross_section_barn: sigma_b,
                },
            ],
        }
    }

    fn base_input() -> IrradiationEvidenceInput {
        IrradiationEvidenceInput {
            subject_id: "fixture:Ti50-Nb50".to_string(),
            composition: vec![
                IrradiationCompositionComponent {
                    atomic_number: 22,
                    fraction_ppm: 500_000,
                },
                IrradiationCompositionComponent {
                    atomic_number: 41,
                    fraction_ppm: 500_000,
                },
            ],
            temperature_k: 600.0,
            exposure_s: 10_000.0,
            spectrum_source_id: "fixture-spectrum".to_string(),
            spectrum_artifact_sha256: DIGEST_A.to_string(),
            spectrum: spectrum(),
            displacement_curves: vec![curve(22, 1.0, 2.0), curve(41, 3.0, 4.0)],
        }
    }

    #[test]
    fn composition_resolved_rate_matches_manual_sum() {
        let input = base_input();
        let evidence = calculate_composition_dpa_evidence(&input).unwrap();
        let ti = 0.5 * (2.0e14 * 1.0 + 1.0e14 * 2.0) * 1.0e-24;
        let nb = 0.5 * (2.0e14 * 3.0 + 1.0e14 * 4.0) * 1.0e-24;
        let expected = ti + nb;
        assert!((evidence.dpa_rate_per_s - expected).abs() < 1.0e-20);
        assert!((evidence.total_dpa - expected * input.exposure_s).abs() < 1.0e-16);
    }

    #[test]
    fn changing_non_primary_constituent_changes_the_result() {
        let low_nb = calculate_composition_dpa_evidence(&base_input()).unwrap();
        let mut changed = base_input();
        changed.displacement_curves[1] = curve(41, 30.0, 40.0);
        let high_nb = calculate_composition_dpa_evidence(&changed).unwrap();
        assert!(high_nb.dpa_rate_per_s > low_nb.dpa_rate_per_s * 5.0);
    }

    #[test]
    fn missing_non_primary_element_curve_is_rejected() {
        let mut input = base_input();
        input.displacement_curves.pop();
        assert!(matches!(
            calculate_composition_dpa_evidence(&input),
            Err(IrradiationEvidenceError::CurveCountMismatch { .. })
        ));
    }

    #[test]
    fn mixed_nrt_and_arc_curves_are_rejected() {
        let mut input = base_input();
        input.displacement_curves[1].model = DamageMetricModel::NrtDpa;
        assert_eq!(
            calculate_composition_dpa_evidence(&input),
            Err(IrradiationEvidenceError::MixedDamageModels)
        );
    }

    #[test]
    fn energy_grid_mismatch_is_rejected() {
        let mut input = base_input();
        input.displacement_curves[1].bins[1].energy_upper_ev = 14.0e6;
        assert!(matches!(
            calculate_composition_dpa_evidence(&input),
            Err(IrradiationEvidenceError::EnergyGridMismatch {
                atomic_number: 41,
                index: 1
            })
        ));
    }

    #[test]
    fn spectrum_digest_is_required() {
        let mut input = base_input();
        input.spectrum_artifact_sha256 = "not-a-sha256".to_string();
        assert_eq!(
            calculate_composition_dpa_evidence(&input),
            Err(IrradiationEvidenceError::InvalidSha256 { kind: "spectrum" })
        );
    }

    #[test]
    fn zero_flux_is_valid_and_yields_zero_dpa() {
        let mut input = base_input();
        for bin in &mut input.spectrum {
            bin.flux_n_cm2_s = 0.0;
        }
        let evidence = calculate_composition_dpa_evidence(&input).unwrap();
        assert_eq!(evidence.dpa_rate_per_s, 0.0);
        assert_eq!(evidence.total_dpa, 0.0);
    }
}
