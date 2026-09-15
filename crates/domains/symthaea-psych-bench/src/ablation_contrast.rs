// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit 2x2 design identity for single-mechanism ablation contrasts.
//!
//! This module binds the already-qualified intervention cells into one
//! content-addressed contrast set. It does not estimate an effect and grants no
//! causal/statistical authority by itself.

use crate::ablation_contract::{
    AblationContractError, AblationIntervention, AblationInterventionKind, AblationMechanism,
    LegacyAblationPresetId,
};
use crate::harness::AblationPreset;
use serde::{Deserialize, Serialize};

pub const ABLATION_CONTRAST_SET_SCHEMA_VERSION: &str = "psych-ablation-contrast-set-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AblationContrastCellRole {
    Baseline,
    MechanismOnly,
    NuisanceOnly,
    Combined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AblationContrastCellReceipt {
    pub role: AblationContrastCellRole,
    pub intervention_id: String,
    pub matched_set_id: String,
    pub intervention_kind: AblationInterventionKind,
    pub config_digest: String,
    pub mechanism_absent: bool,
    pub nuisance_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AblationContrastSetReceipt {
    pub schema_version: String,
    pub contrast_set_id: String,
    pub source_preset: LegacyAblationPresetId,
    pub mechanism: AblationMechanism,
    pub seed: u64,
    pub cells: Vec<AblationContrastCellReceipt>,
    pub contrast_digest: String,
}

#[derive(Debug, Clone)]
pub struct AblationContrastSet {
    baseline: AblationIntervention,
    mechanism_only: AblationIntervention,
    nuisance_only: AblationIntervention,
    combined: AblationIntervention,
    receipt: AblationContrastSetReceipt,
}

impl AblationContrastSet {
    /// Build the canonical four-cell design for one qualified single-mechanism
    /// preset at one seed.
    pub fn single_mechanism(
        preset: AblationPreset,
        seed: u64,
    ) -> Result<Self, AblationContrastError> {
        let mechanism = mechanism_for_preset(preset)?;
        let baseline = AblationIntervention::baseline(seed)?;
        let mechanism_only = AblationIntervention::mechanism_only(preset, seed)?;
        let nuisance_only = AblationIntervention::matched_nuisance_only(preset, seed)?;
        let combined = AblationIntervention::legacy_combined(preset, seed)?;

        let receipt = build_receipt(
            preset,
            mechanism,
            seed,
            &baseline,
            &mechanism_only,
            &nuisance_only,
            &combined,
        )?;
        let set = Self {
            baseline,
            mechanism_only,
            nuisance_only,
            combined,
            receipt,
        };
        set.validate()?;
        Ok(set)
    }

    pub fn receipt(&self) -> &AblationContrastSetReceipt {
        &self.receipt
    }

    pub fn baseline(&self) -> &AblationIntervention {
        &self.baseline
    }

    pub fn mechanism_only(&self) -> &AblationIntervention {
        &self.mechanism_only
    }

    pub fn nuisance_only(&self) -> &AblationIntervention {
        &self.nuisance_only
    }

    pub fn combined(&self) -> &AblationIntervention {
        &self.combined
    }

    pub fn validate(&self) -> Result<(), AblationContrastError> {
        self.validate_receipt(&self.receipt)
    }

    /// Recompute the full set identity from the authority-bearing interventions
    /// and require an externally supplied receipt to match exactly.
    pub fn validate_receipt(
        &self,
        supplied: &AblationContrastSetReceipt,
    ) -> Result<(), AblationContrastError> {
        let preset = supplied.source_preset.to_preset();
        let mechanism = mechanism_for_preset(preset)?;
        let expected = build_receipt(
            preset,
            mechanism,
            supplied.seed,
            &self.baseline,
            &self.mechanism_only,
            &self.nuisance_only,
            &self.combined,
        )?;
        if &expected == supplied {
            Ok(())
        } else {
            Err(AblationContrastError::ReceiptMismatch)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AblationContrastError {
    Contract(AblationContractError),
    RequiresSingleMechanismDesign,
    CellKindMismatch { role: AblationContrastCellRole },
    CellFactorMismatch { role: AblationContrastCellRole },
    CellSeedMismatch { role: AblationContrastCellRole },
    CellPresetMismatch { role: AblationContrastCellRole },
    MatchedArmIdentityMismatch,
    ReceiptMismatch,
    Serialization(String),
}

impl From<AblationContractError> for AblationContrastError {
    fn from(value: AblationContractError) -> Self {
        Self::Contract(value)
    }
}

fn mechanism_for_preset(preset: AblationPreset) -> Result<AblationMechanism, AblationContrastError> {
    match preset {
        AblationPreset::NoFep => Ok(AblationMechanism::Fep),
        AblationPreset::NoSocial => Ok(AblationMechanism::Social),
        AblationPreset::ReducedWm => Ok(AblationMechanism::WorkingMemoryCapacity),
        AblationPreset::FullConsciousness | AblationPreset::CfcOnly | AblationPreset::HdcOnly => {
            Err(AblationContrastError::RequiresSingleMechanismDesign)
        }
    }
}

fn build_receipt(
    preset: AblationPreset,
    mechanism: AblationMechanism,
    seed: u64,
    baseline: &AblationIntervention,
    mechanism_only: &AblationIntervention,
    nuisance_only: &AblationIntervention,
    combined: &AblationIntervention,
) -> Result<AblationContrastSetReceipt, AblationContrastError> {
    for intervention in [baseline, mechanism_only, nuisance_only, combined] {
        intervention.validate()?;
    }

    let source_preset = LegacyAblationPresetId::from_preset(preset);
    let baseline_cell = validate_cell(
        baseline,
        seed,
        mechanism,
        ExpectedCell {
            role: AblationContrastCellRole::Baseline,
            source_preset: LegacyAblationPresetId::FullConsciousness,
            kind: AblationInterventionKind::Baseline,
            mechanism_absent: false,
            nuisance_present: false,
        },
    )?;
    let mechanism_cell = validate_cell(
        mechanism_only,
        seed,
        mechanism,
        ExpectedCell {
            role: AblationContrastCellRole::MechanismOnly,
            source_preset,
            kind: AblationInterventionKind::MechanismOnly,
            mechanism_absent: true,
            nuisance_present: false,
        },
    )?;
    let nuisance_cell = validate_cell(
        nuisance_only,
        seed,
        mechanism,
        ExpectedCell {
            role: AblationContrastCellRole::NuisanceOnly,
            source_preset,
            kind: AblationInterventionKind::NuisanceOnly,
            mechanism_absent: false,
            nuisance_present: true,
        },
    )?;
    let combined_cell = validate_cell(
        combined,
        seed,
        mechanism,
        ExpectedCell {
            role: AblationContrastCellRole::Combined,
            source_preset,
            kind: AblationInterventionKind::CombinedModel,
            mechanism_absent: true,
            nuisance_present: true,
        },
    )?;

    let matched = mechanism_only.receipt().matched_set_id.as_str();
    if nuisance_only.receipt().matched_set_id != matched
        || combined.receipt().matched_set_id != matched
    {
        return Err(AblationContrastError::MatchedArmIdentityMismatch);
    }

    let contrast_set_id = format!("{}::factorial::seed-{seed}", source_preset.slug());
    let cells = vec![baseline_cell, mechanism_cell, nuisance_cell, combined_cell];
    let material = ContrastDigestMaterial {
        schema_version: ABLATION_CONTRAST_SET_SCHEMA_VERSION,
        contrast_set_id: &contrast_set_id,
        source_preset,
        mechanism,
        seed,
        cells: &cells,
    };
    let contrast_digest = contrast_digest_hex(&material)?;

    Ok(AblationContrastSetReceipt {
        schema_version: ABLATION_CONTRAST_SET_SCHEMA_VERSION.to_string(),
        contrast_set_id,
        source_preset,
        mechanism,
        seed,
        cells,
        contrast_digest,
    })
}

#[derive(Debug, Clone, Copy)]
struct ExpectedCell {
    role: AblationContrastCellRole,
    source_preset: LegacyAblationPresetId,
    kind: AblationInterventionKind,
    mechanism_absent: bool,
    nuisance_present: bool,
}

fn validate_cell(
    intervention: &AblationIntervention,
    seed: u64,
    mechanism: AblationMechanism,
    expected: ExpectedCell,
) -> Result<AblationContrastCellReceipt, AblationContrastError> {
    let receipt = intervention.receipt();
    if receipt.seed != seed {
        return Err(AblationContrastError::CellSeedMismatch {
            role: expected.role,
        });
    }
    if receipt.source_preset != expected.source_preset {
        return Err(AblationContrastError::CellPresetMismatch {
            role: expected.role,
        });
    }
    if receipt.kind != expected.kind {
        return Err(AblationContrastError::CellKindMismatch {
            role: expected.role,
        });
    }

    let expected_mechanisms = if expected.mechanism_absent {
        vec![mechanism]
    } else {
        Vec::new()
    };
    let has_nuisance = receipt.encoding_noise.to_bits() != 0.0f64.to_bits();
    if receipt.mechanisms != expected_mechanisms || has_nuisance != expected.nuisance_present {
        return Err(AblationContrastError::CellFactorMismatch {
            role: expected.role,
        });
    }

    Ok(AblationContrastCellReceipt {
        role: expected.role,
        intervention_id: receipt.intervention_id.clone(),
        matched_set_id: receipt.matched_set_id.clone(),
        intervention_kind: receipt.kind,
        config_digest: receipt.config_digest.clone(),
        mechanism_absent: expected.mechanism_absent,
        nuisance_present: expected.nuisance_present,
    })
}

#[derive(Serialize)]
struct ContrastDigestMaterial<'a> {
    schema_version: &'a str,
    contrast_set_id: &'a str,
    source_preset: LegacyAblationPresetId,
    mechanism: AblationMechanism,
    seed: u64,
    cells: &'a [AblationContrastCellReceipt],
}

fn contrast_digest_hex(
    material: &ContrastDigestMaterial<'_>,
) -> Result<String, AblationContrastError> {
    let bytes = serde_json::to_vec(material)
        .map_err(|error| AblationContrastError::Serialization(error.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.psych.ablation-contrast-set.v1\0");
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const SINGLE_PRESETS: [AblationPreset; 3] = [
        AblationPreset::NoFep,
        AblationPreset::NoSocial,
        AblationPreset::ReducedWm,
    ];

    #[test]
    fn every_single_mechanism_preset_builds_complete_four_cell_set() {
        for preset in SINGLE_PRESETS {
            let set = AblationContrastSet::single_mechanism(preset, 42).unwrap();
            assert!(set.validate().is_ok());
            assert_eq!(set.receipt().cells.len(), 4);
            let roles = set
                .receipt()
                .cells
                .iter()
                .map(|cell| cell.role)
                .collect::<BTreeSet<_>>();
            assert_eq!(roles.len(), 4);
        }
    }

    #[test]
    fn baseline_is_explicit_member_not_inferred_from_seed() {
        let set = AblationContrastSet::single_mechanism(AblationPreset::NoFep, 42).unwrap();
        let receipt = set.receipt();
        let baseline = &receipt.cells[0];
        let mechanism = &receipt.cells[1];
        assert_eq!(baseline.role, AblationContrastCellRole::Baseline);
        assert_eq!(baseline.intervention_id, "full_consciousness::baseline::seed-42");
        assert_ne!(baseline.matched_set_id, mechanism.matched_set_id);
        assert_eq!(mechanism.matched_set_id, receipt.cells[2].matched_set_id);
        assert_eq!(mechanism.matched_set_id, receipt.cells[3].matched_set_id);
    }

    #[test]
    fn four_cells_bind_expected_factor_states() {
        let set = AblationContrastSet::single_mechanism(AblationPreset::ReducedWm, 7).unwrap();
        let cells = &set.receipt().cells;
        assert_eq!(
            (cells[0].mechanism_absent, cells[0].nuisance_present),
            (false, false)
        );
        assert_eq!(
            (cells[1].mechanism_absent, cells[1].nuisance_present),
            (true, false)
        );
        assert_eq!(
            (cells[2].mechanism_absent, cells[2].nuisance_present),
            (false, true)
        );
        assert_eq!(
            (cells[3].mechanism_absent, cells[3].nuisance_present),
            (true, true)
        );
    }

    #[test]
    fn contrast_digest_changes_with_seed_or_mechanism() {
        let fep_42 = AblationContrastSet::single_mechanism(AblationPreset::NoFep, 42).unwrap();
        let fep_43 = AblationContrastSet::single_mechanism(AblationPreset::NoFep, 43).unwrap();
        let social_42 =
            AblationContrastSet::single_mechanism(AblationPreset::NoSocial, 42).unwrap();
        assert_ne!(
            fep_42.receipt().contrast_digest,
            fep_43.receipt().contrast_digest
        );
        assert_ne!(
            fep_42.receipt().contrast_digest,
            social_42.receipt().contrast_digest
        );
    }

    #[test]
    fn receipt_tampering_fails_revalidation() {
        let set = AblationContrastSet::single_mechanism(AblationPreset::NoFep, 42).unwrap();
        let mut tampered = set.receipt().clone();
        tampered.cells[0].config_digest.push('0');
        assert_eq!(
            set.validate_receipt(&tampered),
            Err(AblationContrastError::ReceiptMismatch)
        );
    }

    #[test]
    fn multi_mechanism_and_baseline_presets_are_refused() {
        for preset in [
            AblationPreset::FullConsciousness,
            AblationPreset::CfcOnly,
            AblationPreset::HdcOnly,
        ] {
            assert!(matches!(
                AblationContrastSet::single_mechanism(preset, 42),
                Err(AblationContrastError::RequiresSingleMechanismDesign)
                    | Err(AblationContrastError::Contract(
                        AblationContractError::RequiresMultiMechanismDesign
                    ))
                    | Err(AblationContrastError::Contract(
                        AblationContractError::BaselineIsNotAblation
                    ))
            ));
        }
    }

    #[test]
    fn receipt_round_trip_is_descriptive_not_authority() {
        let set = AblationContrastSet::single_mechanism(AblationPreset::NoSocial, 99).unwrap();
        let json = serde_json::to_string(set.receipt()).unwrap();
        let decoded: AblationContrastSetReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(&decoded, set.receipt());
        assert!(set.validate_receipt(&decoded).is_ok());
    }
}
