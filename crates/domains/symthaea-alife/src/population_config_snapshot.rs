// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bit-exact persistence contract for population-level evolutionary policy.
//!
//! `PopulationConfig` is causal execution state: its thresholds decide removal/reproduction,
//! mutation parameters alter offspring genomes, inheritance mode changes genome-source selection,
//! and `organism_cfg` supplies every non-heritable newborn parameter. A population checkpoint that
//! restores organisms but silently rebuilds this policy from defaults can therefore diverge on the
//! very next birth or death.
//!
//! Floating-point values are stored as IEEE-754 bits. This contract preserves representation exactly
//! (including signed zero) and rejects non-finite values before persisted configuration can become
//! executable restore authority.

use serde::{Deserialize, Serialize};

use crate::{InheritanceMode, OrganismConfig, PopulationConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritanceModeSnapshotV1 {
    FromParent,
    RandomPeer,
}

impl From<InheritanceMode> for InheritanceModeSnapshotV1 {
    fn from(value: InheritanceMode) -> Self {
        match value {
            InheritanceMode::FromParent => Self::FromParent,
            InheritanceMode::RandomPeer => Self::RandomPeer,
        }
    }
}

impl InheritanceModeSnapshotV1 {
    pub fn inheritance_mode(self) -> InheritanceMode {
        match self {
            Self::FromParent => InheritanceMode::FromParent,
            Self::RandomPeer => InheritanceMode::RandomPeer,
        }
    }
}

/// Full bit-exact newborn template owned by `PopulationConfig`.
///
/// This intentionally duplicates the persistence shape of `OrganismConfigSnapshotV1` rather than
/// treating one live organism as the population template. Heritable fields may differ across living
/// organisms after mutation, while future offspring still inherit non-heritable fields from this
/// population-owned template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationOrganismTemplateSnapshotV1 {
    set_point_bits: u64,
    metabolic_cost_bits: u64,
    forage_activity_cost_bits: u64,
    forage_efficiency_bits: u64,
    goal_precision_bits: u64,
    effective_temperature_bits: u64,
    dissipation_rate_bits: u64,
    death_energy_threshold_bits: u64,
    action_temperature_bits: u64,
    perceptual_grain_bits: Option<u64>,
    spoilage_sigma_bits: Option<u64>,
    resource_preference_bits: u64,
    resource_prior_bits: u64,
    social_enabled: bool,
    transfer_quantum_bits: u64,
}

impl PopulationOrganismTemplateSnapshotV1 {
    pub fn from_config(config: OrganismConfig) -> Self {
        Self {
            set_point_bits: config.set_point.to_bits(),
            metabolic_cost_bits: config.metabolic_cost.to_bits(),
            forage_activity_cost_bits: config.forage_activity_cost.to_bits(),
            forage_efficiency_bits: config.forage_efficiency.to_bits(),
            goal_precision_bits: config.goal_precision.to_bits(),
            effective_temperature_bits: config.effective_temperature.to_bits(),
            dissipation_rate_bits: config.dissipation_rate.to_bits(),
            death_energy_threshold_bits: config.death_energy_threshold.to_bits(),
            action_temperature_bits: config.action_temperature.to_bits(),
            perceptual_grain_bits: config.perceptual_grain.map(f64::to_bits),
            spoilage_sigma_bits: config.spoilage_sigma.map(f64::to_bits),
            resource_preference_bits: config.resource_preference.to_bits(),
            resource_prior_bits: config.resource_prior.to_bits(),
            social_enabled: config.social_enabled,
            transfer_quantum_bits: config.transfer_quantum.to_bits(),
        }
    }

    fn validate(self) -> Result<OrganismConfig, PopulationConfigSnapshotErrorV1> {
        let config = OrganismConfig {
            set_point: finite("organism_cfg.set_point", self.set_point_bits)?,
            metabolic_cost: finite("organism_cfg.metabolic_cost", self.metabolic_cost_bits)?,
            forage_activity_cost: finite(
                "organism_cfg.forage_activity_cost",
                self.forage_activity_cost_bits,
            )?,
            forage_efficiency: finite(
                "organism_cfg.forage_efficiency",
                self.forage_efficiency_bits,
            )?,
            goal_precision: finite("organism_cfg.goal_precision", self.goal_precision_bits)?,
            effective_temperature: finite(
                "organism_cfg.effective_temperature",
                self.effective_temperature_bits,
            )?,
            dissipation_rate: finite(
                "organism_cfg.dissipation_rate",
                self.dissipation_rate_bits,
            )?,
            death_energy_threshold: finite(
                "organism_cfg.death_energy_threshold",
                self.death_energy_threshold_bits,
            )?,
            action_temperature: finite(
                "organism_cfg.action_temperature",
                self.action_temperature_bits,
            )?,
            perceptual_grain: optional_finite(
                "organism_cfg.perceptual_grain",
                self.perceptual_grain_bits,
            )?,
            spoilage_sigma: optional_finite(
                "organism_cfg.spoilage_sigma",
                self.spoilage_sigma_bits,
            )?,
            resource_preference: finite(
                "organism_cfg.resource_preference",
                self.resource_preference_bits,
            )?,
            resource_prior: finite(
                "organism_cfg.resource_prior",
                self.resource_prior_bits,
            )?,
            social_enabled: self.social_enabled,
            transfer_quantum: finite(
                "organism_cfg.transfer_quantum",
                self.transfer_quantum_bits,
            )?,
        };

        // ActiveInferenceAgent requires a positive action temperature. Preserve the same restore
        // boundary used by OrganismSnapshotV1 rather than allowing population policy to authorize a
        // newborn template that the nested cognitive snapshot contract would reject.
        if config.action_temperature <= 0.0 {
            return Err(PopulationConfigSnapshotErrorV1::NonPositive {
                field: "organism_cfg.action_temperature",
            });
        }
        Ok(config)
    }
}

/// Raw serializable population policy. Deserialization alone does not confer execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationConfigSnapshotV1 {
    death_energy_threshold_bits: u64,
    reproduction_energy_threshold_bits: u64,
    reproduction_energy_cost_bits: u64,
    organism_cfg: PopulationOrganismTemplateSnapshotV1,
    mutation_rate_bits: u64,
    mutation_std_bits: u64,
    inheritance: InheritanceModeSnapshotV1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedPopulationConfigSnapshotV1 {
    config: PopulationConfig,
    snapshot: PopulationConfigSnapshotV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationConfigSnapshotErrorV1 {
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
}

impl PopulationConfigSnapshotV1 {
    pub fn from_config(config: PopulationConfig) -> Self {
        Self {
            death_energy_threshold_bits: config.death_energy_threshold.to_bits(),
            reproduction_energy_threshold_bits: config.reproduction_energy_threshold.to_bits(),
            reproduction_energy_cost_bits: config.reproduction_energy_cost.to_bits(),
            organism_cfg: PopulationOrganismTemplateSnapshotV1::from_config(config.organism_cfg),
            mutation_rate_bits: config.mutation_rate.to_bits(),
            mutation_std_bits: config.mutation_std.to_bits(),
            inheritance: config.inheritance.into(),
        }
    }

    pub fn validate(
        self,
    ) -> Result<ValidatedPopulationConfigSnapshotV1, PopulationConfigSnapshotErrorV1> {
        let config = PopulationConfig {
            death_energy_threshold: finite(
                "death_energy_threshold",
                self.death_energy_threshold_bits,
            )?,
            reproduction_energy_threshold: finite(
                "reproduction_energy_threshold",
                self.reproduction_energy_threshold_bits,
            )?,
            reproduction_energy_cost: finite(
                "reproduction_energy_cost",
                self.reproduction_energy_cost_bits,
            )?,
            organism_cfg: self.organism_cfg.validate()?,
            mutation_rate: finite("mutation_rate", self.mutation_rate_bits)?,
            mutation_std: finite("mutation_std", self.mutation_std_bits)?,
            inheritance: self.inheritance.inheritance_mode(),
        };
        Ok(ValidatedPopulationConfigSnapshotV1 {
            config,
            snapshot: self,
        })
    }
}

impl ValidatedPopulationConfigSnapshotV1 {
    pub fn config(&self) -> PopulationConfig {
        self.config
    }

    pub fn as_snapshot(&self) -> &PopulationConfigSnapshotV1 {
        &self.snapshot
    }

    pub fn into_snapshot(self) -> PopulationConfigSnapshotV1 {
        self.snapshot
    }
}

fn finite(field: &'static str, bits: u64) -> Result<f64, PopulationConfigSnapshotErrorV1> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(PopulationConfigSnapshotErrorV1::NonFinite { field });
    }
    Ok(value)
}

fn optional_finite(
    field: &'static str,
    bits: Option<u64>,
) -> Result<Option<f64>, PopulationConfigSnapshotErrorV1> {
    bits.map(|bits| finite(field, bits)).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_bits_equal(a: PopulationConfig, b: PopulationConfig) {
        assert_eq!(a.death_energy_threshold.to_bits(), b.death_energy_threshold.to_bits());
        assert_eq!(
            a.reproduction_energy_threshold.to_bits(),
            b.reproduction_energy_threshold.to_bits()
        );
        assert_eq!(a.reproduction_energy_cost.to_bits(), b.reproduction_energy_cost.to_bits());
        assert_eq!(a.mutation_rate.to_bits(), b.mutation_rate.to_bits());
        assert_eq!(a.mutation_std.to_bits(), b.mutation_std.to_bits());
        assert_eq!(a.inheritance, b.inheritance);

        let a = PopulationOrganismTemplateSnapshotV1::from_config(a.organism_cfg);
        let b = PopulationOrganismTemplateSnapshotV1::from_config(b.organism_cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn json_round_trip_preserves_complete_population_policy_bit_exactly() {
        let original = PopulationConfig {
            death_energy_threshold: -0.0,
            reproduction_energy_threshold: 0.8125,
            reproduction_energy_cost: 0.375,
            mutation_rate: 0.27,
            mutation_std: 0.03125,
            inheritance: InheritanceMode::RandomPeer,
            organism_cfg: OrganismConfig {
                set_point: -0.0,
                metabolic_cost: 0.015625,
                forage_activity_cost: 0.03125,
                forage_efficiency: 0.73,
                goal_precision: 1.7,
                effective_temperature: 2.25,
                dissipation_rate: 0.0007,
                death_energy_threshold: 0.021,
                action_temperature: 0.63,
                perceptual_grain: Some(0.125),
                spoilage_sigma: Some(-0.0),
                resource_preference: 0.88,
                resource_prior: 0.17,
                social_enabled: true,
                transfer_quantum: 0.042,
            },
        };

        let encoded = serde_json::to_string(&PopulationConfigSnapshotV1::from_config(original))
            .expect("serialize population config snapshot");
        let raw: PopulationConfigSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize population config snapshot");
        let restored = raw.validate().expect("validate population policy").config();
        assert_float_bits_equal(original, restored);
    }

    #[test]
    fn inheritance_mode_round_trips_without_reinterpretation() {
        assert_eq!(
            InheritanceModeSnapshotV1::from(InheritanceMode::FromParent).inheritance_mode(),
            InheritanceMode::FromParent
        );
        assert_eq!(
            InheritanceModeSnapshotV1::from(InheritanceMode::RandomPeer).inheritance_mode(),
            InheritanceMode::RandomPeer
        );
    }

    #[test]
    fn non_finite_population_policy_fails_closed() {
        let mut raw = PopulationConfigSnapshotV1::from_config(PopulationConfig {
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        });
        raw.mutation_rate_bits = f64::NAN.to_bits();
        assert_eq!(
            raw.validate().unwrap_err(),
            PopulationConfigSnapshotErrorV1::NonFinite {
                field: "mutation_rate"
            }
        );
    }

    #[test]
    fn invalid_newborn_action_temperature_cannot_become_restore_authority() {
        let mut cfg = PopulationConfig {
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        };
        cfg.organism_cfg.action_temperature = 0.0;
        assert_eq!(
            PopulationConfigSnapshotV1::from_config(cfg)
                .validate()
                .unwrap_err(),
            PopulationConfigSnapshotErrorV1::NonPositive {
                field: "organism_cfg.action_temperature"
            }
        );
    }
}
