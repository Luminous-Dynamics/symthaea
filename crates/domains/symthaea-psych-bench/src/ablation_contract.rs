// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed intervention semantics for generic psych-bench ablations.
//!
//! Historical `AblationPreset::to_config()` remains unchanged. This module
//! names its combined mechanism+noise conditions explicitly and adds clean
//! single-mechanism and matched nuisance-only controls.

use crate::harness::config::{AblationPreset, BenchmarkConfig};
use serde::{Deserialize, Serialize};

pub const ABLATION_INTERVENTION_SCHEMA_VERSION: &str = "psych-ablation-intervention-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AblationInterventionKind {
    Baseline,
    MechanismOnly,
    NuisanceOnly,
    CombinedModel,
    MultiMechanismCombined,
}

impl AblationInterventionKind {
    const fn slug(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::MechanismOnly => "mechanism_only",
            Self::NuisanceOnly => "nuisance_only",
            Self::CombinedModel => "combined_model",
            Self::MultiMechanismCombined => "multi_mechanism_combined",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AblationMechanism {
    Fep,
    Social,
    WorkingMemoryCapacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyAblationPresetId {
    FullConsciousness,
    CfcOnly,
    NoFep,
    NoSocial,
    ReducedWm,
    HdcOnly,
}

impl LegacyAblationPresetId {
    pub const fn from_preset(preset: AblationPreset) -> Self {
        match preset {
            AblationPreset::FullConsciousness => Self::FullConsciousness,
            AblationPreset::CfcOnly => Self::CfcOnly,
            AblationPreset::NoFep => Self::NoFep,
            AblationPreset::NoSocial => Self::NoSocial,
            AblationPreset::ReducedWm => Self::ReducedWm,
            AblationPreset::HdcOnly => Self::HdcOnly,
        }
    }

    pub const fn to_preset(self) -> AblationPreset {
        match self {
            Self::FullConsciousness => AblationPreset::FullConsciousness,
            Self::CfcOnly => AblationPreset::CfcOnly,
            Self::NoFep => AblationPreset::NoFep,
            Self::NoSocial => AblationPreset::NoSocial,
            Self::ReducedWm => AblationPreset::ReducedWm,
            Self::HdcOnly => AblationPreset::HdcOnly,
        }
    }

    pub const fn slug(self) -> &'static str {
        match self {
            Self::FullConsciousness => "full_consciousness",
            Self::CfcOnly => "cfc_only",
            Self::NoFep => "no_fep",
            Self::NoSocial => "no_social",
            Self::ReducedWm => "reduced_wm",
            Self::HdcOnly => "hdc_only",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationInterventionReceipt {
    pub schema_version: String,
    pub intervention_id: String,
    pub matched_set_id: String,
    pub kind: AblationInterventionKind,
    pub source_preset: LegacyAblationPresetId,
    pub seed: u64,
    pub mechanisms: Vec<AblationMechanism>,
    pub enable_fep: bool,
    pub enable_social: bool,
    pub working_memory_capacity: usize,
    pub encoding_noise: f64,
    pub time_pressure: f64,
    pub config_digest: String,
}

impl AblationInterventionReceipt {
    pub fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AblationContractError> {
        if self.schema_version != ABLATION_INTERVENTION_SCHEMA_VERSION {
            return Err(AblationContractError::UnsupportedSchema);
        }
        if self.intervention_id.trim().is_empty() || self.matched_set_id.trim().is_empty() {
            return Err(AblationContractError::EmptyIdentity);
        }
        if !self.encoding_noise.is_finite() || !(0.0..=1.0).contains(&self.encoding_noise) {
            return Err(AblationContractError::InvalidEncodingNoise);
        }
        if self.time_pressure != 0.0 {
            return Err(AblationContractError::UnexpectedTimePressure);
        }
        if config.seed != self.seed {
            return Err(AblationContractError::SeedMismatch);
        }

        let expected_mechanisms = mechanisms_for_config(config);
        if self.mechanisms != expected_mechanisms {
            return Err(AblationContractError::MechanismMetadataMismatch);
        }
        if config.enable_fep != self.enable_fep
            || config.enable_social != self.enable_social
            || config.working_memory_capacity != self.working_memory_capacity
            || config.encoding_noise.to_bits() != self.encoding_noise.to_bits()
            || config.time_pressure.to_bits() != self.time_pressure.to_bits()
        {
            return Err(AblationContractError::ConfigMetadataMismatch);
        }

        validate_kind_semantics(self.kind, &self.mechanisms, self.encoding_noise)?;
        validate_source_preset_semantics(self)?;

        let expected = reconstruct_config(self);
        if serde_json::to_value(config).map_err(serialization_error)?
            != serde_json::to_value(&expected).map_err(serialization_error)?
        {
            return Err(AblationContractError::UnexpectedConfigDrift);
        }

        if config_digest_hex(config)? != self.config_digest {
            return Err(AblationContractError::ConfigDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AblationIntervention {
    receipt: AblationInterventionReceipt,
    config: BenchmarkConfig,
}

impl AblationIntervention {
    pub fn receipt(&self) -> &AblationInterventionReceipt {
        &self.receipt
    }

    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    pub fn validate(&self) -> Result<(), AblationContractError> {
        self.receipt.validate_config(&self.config)
    }

    pub fn baseline(seed: u64) -> Result<Self, AblationContractError> {
        build_intervention(
            AblationPreset::FullConsciousness,
            seed,
            AblationInterventionKind::Baseline,
            |_| {},
        )
    }

    /// Reproduce the historical `AblationPreset::to_config()` condition exactly,
    /// but label its causal structure explicitly.
    pub fn legacy_combined(
        preset: AblationPreset,
        seed: u64,
    ) -> Result<Self, AblationContractError> {
        let legacy = preset.to_config(seed).base;
        let mechanisms = mechanisms_for_config(&legacy);
        let kind = if mechanisms.is_empty() {
            AblationInterventionKind::Baseline
        } else if mechanisms.len() == 1 {
            AblationInterventionKind::CombinedModel
        } else {
            AblationInterventionKind::MultiMechanismCombined
        };
        build_from_config(preset, seed, kind, legacy)
    }

    /// Change only one intended mechanism axis. Encoding noise remains at the
    /// baseline value so nuisance degradation is not bundled with the mechanism.
    pub fn mechanism_only(
        preset: AblationPreset,
        seed: u64,
    ) -> Result<Self, AblationContractError> {
        require_single_mechanism_preset(preset)?;
        build_intervention(
            preset,
            seed,
            AblationInterventionKind::MechanismOnly,
            |config| apply_mechanism_only(preset, config),
        )
    }

    /// Apply exactly the historical preset's encoding-noise nuisance while
    /// leaving FEP/social/WM mechanisms at baseline values.
    pub fn matched_nuisance_only(
        preset: AblationPreset,
        seed: u64,
    ) -> Result<Self, AblationContractError> {
        require_single_mechanism_preset(preset)?;
        let noise = preset.to_config(seed).base.encoding_noise;
        build_intervention(
            preset,
            seed,
            AblationInterventionKind::NuisanceOnly,
            |config| config.encoding_noise = noise,
        )
    }

    /// Three paired arms with one seed and one matched-set identity:
    /// mechanism-only, nuisance-only, and exact historical combined model.
    pub fn paired_single_mechanism(
        preset: AblationPreset,
        seed: u64,
    ) -> Result<[Self; 3], AblationContractError> {
        require_single_mechanism_preset(preset)?;
        Ok([
            Self::mechanism_only(preset, seed)?,
            Self::matched_nuisance_only(preset, seed)?,
            Self::legacy_combined(preset, seed)?,
        ])
    }
}

fn build_intervention<F>(
    preset: AblationPreset,
    seed: u64,
    kind: AblationInterventionKind,
    mutate: F,
) -> Result<AblationIntervention, AblationContractError>
where
    F: FnOnce(&mut BenchmarkConfig),
{
    let mut config = BenchmarkConfig {
        seed,
        ..Default::default()
    };
    mutate(&mut config);
    build_from_config(preset, seed, kind, config)
}

fn build_from_config(
    preset: AblationPreset,
    seed: u64,
    kind: AblationInterventionKind,
    config: BenchmarkConfig,
) -> Result<AblationIntervention, AblationContractError> {
    let source_preset = LegacyAblationPresetId::from_preset(preset);
    let matched_set_id = format!("{}::seed-{seed}", source_preset.slug());
    let intervention_id = format!(
        "{}::{}::seed-{seed}",
        source_preset.slug(),
        kind.slug()
    );
    let receipt = AblationInterventionReceipt {
        schema_version: ABLATION_INTERVENTION_SCHEMA_VERSION.to_string(),
        intervention_id,
        matched_set_id,
        kind,
        source_preset,
        seed,
        mechanisms: mechanisms_for_config(&config),
        enable_fep: config.enable_fep,
        enable_social: config.enable_social,
        working_memory_capacity: config.working_memory_capacity,
        encoding_noise: config.encoding_noise,
        time_pressure: config.time_pressure,
        config_digest: config_digest_hex(&config)?,
    };
    let intervention = AblationIntervention { receipt, config };
    intervention.validate()?;
    Ok(intervention)
}

fn require_single_mechanism_preset(preset: AblationPreset) -> Result<(), AblationContractError> {
    match preset {
        AblationPreset::NoFep | AblationPreset::NoSocial | AblationPreset::ReducedWm => Ok(()),
        AblationPreset::FullConsciousness => Err(AblationContractError::BaselineIsNotAblation),
        AblationPreset::CfcOnly | AblationPreset::HdcOnly => {
            Err(AblationContractError::RequiresMultiMechanismDesign)
        }
    }
}

fn apply_mechanism_only(preset: AblationPreset, config: &mut BenchmarkConfig) {
    match preset {
        AblationPreset::NoFep => config.enable_fep = false,
        AblationPreset::NoSocial => config.enable_social = false,
        AblationPreset::ReducedWm => config.working_memory_capacity = 3,
        AblationPreset::FullConsciousness | AblationPreset::CfcOnly | AblationPreset::HdcOnly => {}
    }
}

fn mechanisms_for_config(config: &BenchmarkConfig) -> Vec<AblationMechanism> {
    let baseline = BenchmarkConfig::default();
    let mut mechanisms = Vec::new();
    if config.enable_fep != baseline.enable_fep {
        mechanisms.push(AblationMechanism::Fep);
    }
    if config.enable_social != baseline.enable_social {
        mechanisms.push(AblationMechanism::Social);
    }
    if config.working_memory_capacity != baseline.working_memory_capacity {
        mechanisms.push(AblationMechanism::WorkingMemoryCapacity);
    }
    mechanisms
}

fn validate_kind_semantics(
    kind: AblationInterventionKind,
    mechanisms: &[AblationMechanism],
    encoding_noise: f64,
) -> Result<(), AblationContractError> {
    let has_noise = encoding_noise.to_bits() != 0.0f64.to_bits();
    match kind {
        AblationInterventionKind::Baseline if mechanisms.is_empty() && !has_noise => Ok(()),
        AblationInterventionKind::MechanismOnly if mechanisms.len() == 1 && !has_noise => Ok(()),
        AblationInterventionKind::NuisanceOnly if mechanisms.is_empty() && has_noise => Ok(()),
        AblationInterventionKind::CombinedModel if mechanisms.len() == 1 && has_noise => Ok(()),
        AblationInterventionKind::MultiMechanismCombined
            if mechanisms.len() > 1 && has_noise =>
        {
            Ok(())
        }
        _ => Err(AblationContractError::InterventionKindMismatch),
    }
}

fn validate_source_preset_semantics(
    receipt: &AblationInterventionReceipt,
) -> Result<(), AblationContractError> {
    let preset = receipt.source_preset.to_preset();
    let legacy = preset.to_config(receipt.seed).base;
    let legacy_mechanisms = mechanisms_for_config(&legacy);
    let same_noise = receipt.encoding_noise.to_bits() == legacy.encoding_noise.to_bits();

    let valid = match receipt.kind {
        AblationInterventionKind::Baseline => {
            preset == AblationPreset::FullConsciousness
                && receipt.mechanisms.is_empty()
                && receipt.encoding_noise.to_bits() == 0.0f64.to_bits()
        }
        AblationInterventionKind::MechanismOnly => {
            legacy_mechanisms.len() == 1
                && receipt.mechanisms == legacy_mechanisms
                && receipt.encoding_noise.to_bits() == 0.0f64.to_bits()
        }
        AblationInterventionKind::NuisanceOnly => {
            legacy_mechanisms.len() == 1 && receipt.mechanisms.is_empty() && same_noise
        }
        AblationInterventionKind::CombinedModel => {
            legacy_mechanisms.len() == 1 && receipt.mechanisms == legacy_mechanisms && same_noise
        }
        AblationInterventionKind::MultiMechanismCombined => {
            legacy_mechanisms.len() > 1 && receipt.mechanisms == legacy_mechanisms && same_noise
        }
    };

    if valid {
        Ok(())
    } else {
        Err(AblationContractError::SourcePresetMismatch)
    }
}

fn reconstruct_config(receipt: &AblationInterventionReceipt) -> BenchmarkConfig {
    let mut config = BenchmarkConfig {
        seed: receipt.seed,
        ..Default::default()
    };
    config.enable_fep = receipt.enable_fep;
    config.enable_social = receipt.enable_social;
    config.working_memory_capacity = receipt.working_memory_capacity;
    config.encoding_noise = receipt.encoding_noise;
    config.time_pressure = receipt.time_pressure;
    config
}

fn config_digest_hex(config: &BenchmarkConfig) -> Result<String, AblationContractError> {
    let bytes = serde_json::to_vec(config).map_err(serialization_error)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.psych.ablation-config.v1\0");
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn serialization_error(error: serde_json::Error) -> AblationContractError {
    AblationContractError::Serialization(error.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AblationContractError {
    UnsupportedSchema,
    EmptyIdentity,
    InvalidEncodingNoise,
    UnexpectedTimePressure,
    SeedMismatch,
    MechanismMetadataMismatch,
    ConfigMetadataMismatch,
    UnexpectedConfigDrift,
    ConfigDigestMismatch,
    InterventionKindMismatch,
    SourcePresetMismatch,
    BaselineIsNotAblation,
    RequiresMultiMechanismDesign,
    Serialization(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_json_equal(left: &BenchmarkConfig, right: &BenchmarkConfig) {
        assert_eq!(
            serde_json::to_value(left).unwrap(),
            serde_json::to_value(right).unwrap()
        );
    }

    #[test]
    fn legacy_combined_reproduces_every_historical_preset_exactly() {
        for &preset in AblationPreset::all() {
            let legacy = preset.to_config(42).base;
            let typed = AblationIntervention::legacy_combined(preset, 42).unwrap();
            assert_json_equal(typed.config(), &legacy);
            assert!(typed.validate().is_ok());
        }
    }

    #[test]
    fn no_fep_has_three_distinct_paired_arms_with_one_seed() {
        let [mechanism, nuisance, combined] =
            AblationIntervention::paired_single_mechanism(AblationPreset::NoFep, 77).unwrap();

        assert_eq!(
            mechanism.receipt.kind,
            AblationInterventionKind::MechanismOnly
        );
        assert!(!mechanism.config.enable_fep);
        assert_eq!(mechanism.config.encoding_noise.to_bits(), 0.0f64.to_bits());

        assert_eq!(
            nuisance.receipt.kind,
            AblationInterventionKind::NuisanceOnly
        );
        assert!(nuisance.config.enable_fep);
        assert_eq!(nuisance.config.encoding_noise.to_bits(), 0.25f64.to_bits());

        assert_eq!(
            combined.receipt.kind,
            AblationInterventionKind::CombinedModel
        );
        assert!(!combined.config.enable_fep);
        assert_eq!(combined.config.encoding_noise.to_bits(), 0.25f64.to_bits());

        assert_eq!(mechanism.receipt.seed, 77);
        assert_eq!(
            mechanism.receipt.matched_set_id,
            nuisance.receipt.matched_set_id
        );
        assert_eq!(
            mechanism.receipt.matched_set_id,
            combined.receipt.matched_set_id
        );
    }

    #[test]
    fn no_social_and_reduced_wm_keep_noise_out_of_mechanism_arm() {
        let social = AblationIntervention::mechanism_only(AblationPreset::NoSocial, 42).unwrap();
        assert!(!social.config.enable_social);
        assert_eq!(social.config.encoding_noise.to_bits(), 0.0f64.to_bits());

        let wm = AblationIntervention::mechanism_only(AblationPreset::ReducedWm, 42).unwrap();
        assert_eq!(wm.config.working_memory_capacity, 3);
        assert_eq!(wm.config.encoding_noise.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn multi_mechanism_presets_cannot_masquerade_as_single_mechanism_arms() {
        for preset in [AblationPreset::CfcOnly, AblationPreset::HdcOnly] {
            assert!(matches!(
                AblationIntervention::mechanism_only(preset, 42),
                Err(AblationContractError::RequiresMultiMechanismDesign)
            ));
            let combined = AblationIntervention::legacy_combined(preset, 42).unwrap();
            assert_eq!(
                combined.receipt.kind,
                AblationInterventionKind::MultiMechanismCombined
            );
            assert!(combined.receipt.mechanisms.len() > 1);
        }
    }

    #[test]
    fn combined_condition_cannot_be_relabelled_mechanism_only() {
        let combined = AblationIntervention::legacy_combined(AblationPreset::NoFep, 42).unwrap();
        let mut receipt = combined.receipt.clone();
        receipt.kind = AblationInterventionKind::MechanismOnly;
        assert_eq!(
            receipt.validate_config(combined.config()),
            Err(AblationContractError::InterventionKindMismatch)
        );
    }

    #[test]
    fn source_preset_cannot_be_relabelled() {
        let mechanism = AblationIntervention::mechanism_only(AblationPreset::NoSocial, 42).unwrap();
        let mut receipt = mechanism.receipt.clone();
        receipt.source_preset = LegacyAblationPresetId::NoFep;
        assert_eq!(
            receipt.validate_config(mechanism.config()),
            Err(AblationContractError::SourcePresetMismatch)
        );
    }

    #[test]
    fn matched_noise_amount_is_bound_to_source_preset() {
        let nuisance =
            AblationIntervention::matched_nuisance_only(AblationPreset::NoFep, 42).unwrap();
        let mut receipt = nuisance.receipt.clone();
        let mut config = nuisance.config.clone();
        receipt.encoding_noise = 0.15;
        config.encoding_noise = 0.15;
        receipt.config_digest = config_digest_hex(&config).unwrap();
        assert_eq!(
            receipt.validate_config(&config),
            Err(AblationContractError::SourcePresetMismatch)
        );
    }

    #[test]
    fn hidden_config_drift_fails_even_when_ablation_metadata_is_unchanged() {
        let intervention = AblationIntervention::mechanism_only(AblationPreset::NoFep, 42).unwrap();
        let mut config = intervention.config.clone();
        config.difficulty = 0.5;
        assert_eq!(
            intervention.receipt.validate_config(&config),
            Err(AblationContractError::UnexpectedConfigDrift)
        );
    }

    #[test]
    fn nuisance_only_control_preserves_all_mechanisms() {
        for preset in [
            AblationPreset::NoFep,
            AblationPreset::NoSocial,
            AblationPreset::ReducedWm,
        ] {
            let nuisance = AblationIntervention::matched_nuisance_only(preset, 42).unwrap();
            assert!(nuisance.receipt.mechanisms.is_empty());
            assert_eq!(
                nuisance.config.enable_fep,
                BenchmarkConfig::default().enable_fep
            );
            assert_eq!(
                nuisance.config.enable_social,
                BenchmarkConfig::default().enable_social
            );
            assert_eq!(
                nuisance.config.working_memory_capacity,
                BenchmarkConfig::default().working_memory_capacity
            );
            assert!(nuisance.config.encoding_noise > 0.0);
        }
    }

    #[test]
    fn tampered_config_digest_fails_validation() {
        let intervention = AblationIntervention::mechanism_only(AblationPreset::NoFep, 42).unwrap();
        let mut receipt = intervention.receipt.clone();
        receipt.config_digest = "0".repeat(64);
        assert_eq!(
            receipt.validate_config(intervention.config()),
            Err(AblationContractError::ConfigDigestMismatch)
        );
    }
}
