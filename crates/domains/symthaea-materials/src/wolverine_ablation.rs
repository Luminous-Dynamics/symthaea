// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic Ti-Zr-Nb-Ta-Er ablation-study manifest.
//!
//! This module turns the historical "Wolverine" alloy idea into a controlled
//! composition study.  The causal sweep starts from equiatomic Ti-Zr-Nb-Ta and
//! substitutes Er while diluting all four parent elements equally.  The legacy
//! hard-coded Ti25-Zr25-Nb20-Ta15-Er15 composition is retained as a separate
//! reference point so it cannot be confused with a member of the causal sweep.
//!
//! Compositions are represented as integer parts-per-million atomic fractions.
//! That makes the generated manifest byte-for-byte reproducible and avoids
//! floating-point formatting becoming part of candidate identity.
//!
//! Descriptor outputs produced here remain MAT-003 screening telemetry only.
//! They do not establish phase stability, novelty, radiation tolerance,
//! synthesis, or experimental validation.

use crate::hea_screening::{
    HeaElementDescriptor, HeaScreeningDescriptors, HeaScreeningError,
    calculate_hea_descriptors,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Integer denominator used for all manifest atomic fractions.
pub const COMPOSITION_PPM_TOTAL: u32 = 1_000_000;

/// Er loadings in the controlled causal sweep: 0, 0.5, 1, 2, 5, 10, 15 at.%.
pub const CONTROLLED_ER_PPM: [u32; 7] = [0, 5_000, 10_000, 20_000, 50_000, 100_000, 150_000];

/// Versioned property table used only for inexpensive MAT-003 telemetry.
pub const WOLVERINE_SCREENING_TABLE_ID: &str =
    "symthaea-wolverine-screening-table-v1:legacy-radii-melting-points";

const TI_Z: u16 = 22;
const ZR_Z: u16 = 40;
const NB_Z: u16 = 41;
const TA_Z: u16 = 73;
const ER_Z: u16 = 68;

/// Role of one candidate in the Wolverine study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WolverineCandidateRole {
    /// Member of the controlled Er-substitution sweep.
    ControlledSweep,
    /// Historical hard-coded Wolverine composition retained only as a reference.
    LegacyReference,
}

/// One exact elemental atomic fraction represented in integer ppm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicFractionPpm {
    /// Atomic number.
    pub atomic_number: u16,
    /// Atomic fraction in parts per million of the whole alloy.
    pub fraction_ppm: u32,
}

/// One reproducibly identified Wolverine-study composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WolverineCandidate {
    /// Canonical composition identity generated from exact integer fractions.
    pub canonical_id: String,
    /// Whether this is part of the causal sweep or the historical reference.
    pub role: WolverineCandidateRole,
    /// Exact Er fraction in ppm.
    pub er_fraction_ppm: u32,
    /// Positive elemental fractions in canonical Ti, Zr, Nb, Ta, Er order.
    pub components: Vec<AtomicFractionPpm>,
    /// Versioned property table used when producing cheap descriptor telemetry.
    pub screening_table_id: String,
}

impl WolverineCandidate {
    /// Atomic fraction of an element, if present, as an exact ppm integer.
    pub fn fraction_ppm(&self, atomic_number: u16) -> Option<u32> {
        self.components
            .iter()
            .find(|component| component.atomic_number == atomic_number)
            .map(|component| component.fraction_ppm)
    }

    /// Sum of all positive component fractions.
    pub fn total_fraction_ppm(&self) -> u32 {
        self.components.iter().map(|component| component.fraction_ppm).sum()
    }
}

/// Frozen deterministic manifest for the first Ti-Zr-Nb-Ta-Er ablation campaign.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WolverineAblationManifest {
    /// Schema version for serialization and future migration.
    pub schema_version: u32,
    /// Stable study identifier.
    pub study_id: String,
    /// Generator identity; changing generation semantics requires a new version.
    pub generator_version: String,
    /// Controlled sweep followed by the distinct legacy reference point.
    pub candidates: Vec<WolverineCandidate>,
}

/// Generate the first deterministic Wolverine Er-ablation manifest.
pub fn generate_wolverine_ablation_manifest(
) -> Result<WolverineAblationManifest, WolverineAblationError> {
    let mut candidates = Vec::with_capacity(CONTROLLED_ER_PPM.len() + 1);
    for er_fraction_ppm in CONTROLLED_ER_PPM {
        candidates.push(controlled_candidate(er_fraction_ppm)?);
    }
    candidates.push(legacy_wolverine_candidate());

    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in &candidates {
        validate_candidate(candidate)?;
        if !ids.insert(candidate.canonical_id.clone()) {
            return Err(WolverineAblationError::DuplicateCanonicalId(
                candidate.canonical_id.clone(),
            ));
        }
    }

    Ok(WolverineAblationManifest {
        schema_version: 1,
        study_id: "MAT-004:Ti-Zr-Nb-Ta-Er-ablation-v1".to_string(),
        generator_version: "mat-004-generator-v1".to_string(),
        candidates,
    })
}

/// Produce MAT-003 descriptor telemetry for one frozen candidate.
///
/// No pair-enthalpy table is supplied here because MAT-004 does not yet bind a
/// single sourced Ti/Zr/Nb/Ta/Er interaction table.  Consequently `delta_h_mix`
/// and `omega` remain unknown rather than silently inventing pair interactions.
pub fn screen_wolverine_candidate(
    candidate: &WolverineCandidate,
) -> Result<HeaScreeningDescriptors, WolverineAblationError> {
    validate_candidate(candidate)?;
    let inputs = candidate
        .components
        .iter()
        .map(screening_input)
        .collect::<Result<Vec<_>, _>>()?;
    calculate_hea_descriptors(&inputs, &[]).map_err(WolverineAblationError::Screening)
}

fn controlled_candidate(er_fraction_ppm: u32) -> Result<WolverineCandidate, WolverineAblationError> {
    if er_fraction_ppm > COMPOSITION_PPM_TOTAL {
        return Err(WolverineAblationError::InvalidErFraction(er_fraction_ppm));
    }
    let parent_total = COMPOSITION_PPM_TOTAL - er_fraction_ppm;
    if parent_total % 4 != 0 {
        return Err(WolverineAblationError::ParentFractionNotDivisibleByFour(
            parent_total,
        ));
    }
    let parent_each = parent_total / 4;

    let mut components = vec![
        AtomicFractionPpm {
            atomic_number: TI_Z,
            fraction_ppm: parent_each,
        },
        AtomicFractionPpm {
            atomic_number: ZR_Z,
            fraction_ppm: parent_each,
        },
        AtomicFractionPpm {
            atomic_number: NB_Z,
            fraction_ppm: parent_each,
        },
        AtomicFractionPpm {
            atomic_number: TA_Z,
            fraction_ppm: parent_each,
        },
    ];
    if er_fraction_ppm > 0 {
        components.push(AtomicFractionPpm {
            atomic_number: ER_Z,
            fraction_ppm: er_fraction_ppm,
        });
    }

    Ok(WolverineCandidate {
        canonical_id: canonical_id("sweep", parent_each, parent_each, parent_each, parent_each, er_fraction_ppm),
        role: WolverineCandidateRole::ControlledSweep,
        er_fraction_ppm,
        components,
        screening_table_id: WOLVERINE_SCREENING_TABLE_ID.to_string(),
    })
}

fn legacy_wolverine_candidate() -> WolverineCandidate {
    WolverineCandidate {
        canonical_id: canonical_id("legacy", 250_000, 250_000, 200_000, 150_000, 150_000),
        role: WolverineCandidateRole::LegacyReference,
        er_fraction_ppm: 150_000,
        components: vec![
            AtomicFractionPpm {
                atomic_number: TI_Z,
                fraction_ppm: 250_000,
            },
            AtomicFractionPpm {
                atomic_number: ZR_Z,
                fraction_ppm: 250_000,
            },
            AtomicFractionPpm {
                atomic_number: NB_Z,
                fraction_ppm: 200_000,
            },
            AtomicFractionPpm {
                atomic_number: TA_Z,
                fraction_ppm: 150_000,
            },
            AtomicFractionPpm {
                atomic_number: ER_Z,
                fraction_ppm: 150_000,
            },
        ],
        screening_table_id: WOLVERINE_SCREENING_TABLE_ID.to_string(),
    }
}

fn canonical_id(prefix: &str, ti: u32, zr: u32, nb: u32, ta: u32, er: u32) -> String {
    format!(
        "{prefix}:Ti{ti:06}-Zr{zr:06}-Nb{nb:06}-Ta{ta:06}-Er{er:06}:ppm"
    )
}

fn validate_candidate(candidate: &WolverineCandidate) -> Result<(), WolverineAblationError> {
    if candidate.total_fraction_ppm() != COMPOSITION_PPM_TOTAL {
        return Err(WolverineAblationError::FractionsDoNotSumToTotal {
            canonical_id: candidate.canonical_id.clone(),
            sum: candidate.total_fraction_ppm(),
        });
    }
    let er = candidate.fraction_ppm(ER_Z).unwrap_or(0);
    if er != candidate.er_fraction_ppm {
        return Err(WolverineAblationError::ErFractionMismatch {
            declared: candidate.er_fraction_ppm,
            actual: er,
        });
    }
    let mut seen = HashSet::new();
    for component in &candidate.components {
        if component.fraction_ppm == 0 {
            return Err(WolverineAblationError::ZeroFractionComponent(
                component.atomic_number,
            ));
        }
        if !seen.insert(component.atomic_number) {
            return Err(WolverineAblationError::DuplicateElement(
                component.atomic_number,
            ));
        }
        if !matches!(component.atomic_number, TI_Z | ZR_Z | NB_Z | TA_Z | ER_Z) {
            return Err(WolverineAblationError::UnknownElement(
                component.atomic_number,
            ));
        }
    }
    Ok(())
}

fn screening_input(
    component: &AtomicFractionPpm,
) -> Result<HeaElementDescriptor, WolverineAblationError> {
    let (radius, melting, vec) = match component.atomic_number {
        // Values intentionally match the historical Symthaea HEA table so MAT-004
        // can regression-test the old Wolverine composition before source-table
        // replacement in later evidence tranches.
        TI_Z => (147.0, 1941.0, Some(4.0)),
        ZR_Z => (160.0, 2128.0, Some(4.0)),
        NB_Z => (146.0, 2750.0, Some(5.0)),
        TA_Z => (146.0, 3290.0, Some(5.0)),
        // Er VEC is deliberately left unknown rather than choosing an ambiguous
        // lanthanide convention merely to obtain a phase-tendency scalar.
        ER_Z => (176.0, 1802.0, None),
        other => return Err(WolverineAblationError::UnknownElement(other)),
    };

    Ok(HeaElementDescriptor {
        atomic_number: component.atomic_number,
        atomic_fraction: component.fraction_ppm as f64 / COMPOSITION_PPM_TOTAL as f64,
        atomic_radius_pm: radius,
        melting_point_k: melting,
        vec,
    })
}

/// Deterministic-manifest or screening failure.
#[derive(Debug, Clone, PartialEq)]
pub enum WolverineAblationError {
    /// Requested Er loading exceeded 100 at.%.
    InvalidErFraction(u32),
    /// Parent remainder cannot be divided equally among Ti/Zr/Nb/Ta in ppm.
    ParentFractionNotDivisibleByFour(u32),
    /// Candidate fractions did not sum to exactly one million ppm.
    FractionsDoNotSumToTotal {
        /// Candidate identity.
        canonical_id: String,
        /// Actual integer sum.
        sum: u32,
    },
    /// Declared Er loading disagreed with the composition vector.
    ErFractionMismatch {
        /// Metadata value.
        declared: u32,
        /// Composition value.
        actual: u32,
    },
    /// Zero-valued components are omitted from canonical component vectors.
    ZeroFractionComponent(u16),
    /// An element appeared more than once.
    DuplicateElement(u16),
    /// Element is outside the frozen Ti/Zr/Nb/Ta/Er study space.
    UnknownElement(u16),
    /// Two candidates acquired the same supposedly canonical identity.
    DuplicateCanonicalId(String),
    /// MAT-003 descriptor calculation rejected the candidate.
    Screening(HeaScreeningError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> WolverineAblationManifest {
        generate_wolverine_ablation_manifest().unwrap()
    }

    #[test]
    fn manifest_has_seven_controlled_points_plus_legacy_reference() {
        let manifest = manifest();
        assert_eq!(manifest.candidates.len(), 8);
        assert_eq!(
            manifest
                .candidates
                .iter()
                .filter(|candidate| candidate.role == WolverineCandidateRole::ControlledSweep)
                .count(),
            7
        );
        assert_eq!(
            manifest
                .candidates
                .iter()
                .filter(|candidate| candidate.role == WolverineCandidateRole::LegacyReference)
                .count(),
            1
        );
    }

    #[test]
    fn every_candidate_sums_exactly_to_one_million_ppm() {
        for candidate in manifest().candidates {
            assert_eq!(candidate.total_fraction_ppm(), COMPOSITION_PPM_TOTAL);
        }
    }

    #[test]
    fn controlled_sweep_dilutes_all_four_parent_elements_equally() {
        for candidate in manifest()
            .candidates
            .into_iter()
            .filter(|candidate| candidate.role == WolverineCandidateRole::ControlledSweep)
        {
            let ti = candidate.fraction_ppm(TI_Z).unwrap();
            assert_eq!(candidate.fraction_ppm(ZR_Z), Some(ti));
            assert_eq!(candidate.fraction_ppm(NB_Z), Some(ti));
            assert_eq!(candidate.fraction_ppm(TA_Z), Some(ti));
            assert_eq!(
                ti * 4 + candidate.er_fraction_ppm,
                COMPOSITION_PPM_TOTAL
            );
        }
    }

    #[test]
    fn legacy_wolverine_is_preserved_but_not_misclassified_as_sweep_point() {
        let legacy = manifest()
            .candidates
            .into_iter()
            .find(|candidate| candidate.role == WolverineCandidateRole::LegacyReference)
            .unwrap();
        assert_eq!(legacy.fraction_ppm(TI_Z), Some(250_000));
        assert_eq!(legacy.fraction_ppm(ZR_Z), Some(250_000));
        assert_eq!(legacy.fraction_ppm(NB_Z), Some(200_000));
        assert_eq!(legacy.fraction_ppm(TA_Z), Some(150_000));
        assert_eq!(legacy.fraction_ppm(ER_Z), Some(150_000));
        assert_eq!(legacy.er_fraction_ppm, 150_000);
    }

    #[test]
    fn canonical_ids_are_unique_and_stable() {
        let manifest = manifest();
        let ids: HashSet<_> = manifest
            .candidates
            .iter()
            .map(|candidate| candidate.canonical_id.as_str())
            .collect();
        assert_eq!(ids.len(), manifest.candidates.len());
        assert_eq!(
            manifest.candidates[0].canonical_id,
            "sweep:Ti250000-Zr250000-Nb250000-Ta250000-Er000000:ppm"
        );
        assert_eq!(
            manifest.candidates[6].canonical_id,
            "sweep:Ti212500-Zr212500-Nb212500-Ta212500-Er150000:ppm"
        );
        assert_eq!(
            manifest.candidates[7].canonical_id,
            "legacy:Ti250000-Zr250000-Nb200000-Ta150000-Er150000:ppm"
        );
    }

    #[test]
    fn legacy_descriptor_regression_matches_historical_wolverine_inputs() {
        let legacy = &manifest().candidates[7];
        let descriptors = screen_wolverine_candidate(legacy).unwrap();
        assert!((descriptors.atomic_size_mismatch_pct - 6.988_026_3).abs() < 1.0e-6);
        assert!(
            (descriptors.configurational_entropy_j_mol_k - 13.171_528_55).abs() < 1.0e-6
        );
        assert!(!descriptors.classic_flags.atomic_size_delta_le_6_6_pct);
        assert!(descriptors.classic_flags.entropy_ge_1_5_r);
        assert_eq!(descriptors.vec, None);
        assert_eq!(descriptors.mixing_enthalpy_kj_mol, None);
        assert_eq!(descriptors.omega, None);
    }

    #[test]
    fn size_mismatch_signal_increases_across_the_controlled_er_sweep() {
        let controlled: Vec<_> = manifest()
            .candidates
            .into_iter()
            .filter(|candidate| candidate.role == WolverineCandidateRole::ControlledSweep)
            .collect();
        let deltas: Vec<_> = controlled
            .iter()
            .map(|candidate| {
                screen_wolverine_candidate(candidate)
                    .unwrap()
                    .atomic_size_mismatch_pct
            })
            .collect();
        for window in deltas.windows(2) {
            assert!(window[1] > window[0]);
        }
        assert!(deltas[5] < 6.6, "10 at.% Er remains below the classic screen");
        assert!(deltas[6] > 6.6, "15 at.% Er crosses the classic screen");
    }

    #[test]
    fn er_free_control_has_vec_but_er_points_leave_vec_unknown() {
        let manifest = manifest();
        let control = screen_wolverine_candidate(&manifest.candidates[0]).unwrap();
        assert_eq!(control.vec, Some(4.5));
        for candidate in &manifest.candidates[1..] {
            assert_eq!(screen_wolverine_candidate(candidate).unwrap().vec, None);
        }
    }
}
