// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-actuating semantic model for a compliant human-contact interface stack.
//!
//! The stack is ordered from the human-facing surface inward toward rigid robot
//! structure. It records required mechanical/sensing boundaries but does not
//! select materials, actuator commands, comfort targets, or production safety
//! thresholds. Presence in this model is not qualification evidence.

use std::collections::BTreeSet;

pub const SOMATIC_INTERFACE_STACK_SCHEMA_V1: &str =
    "symthaea.humanoid.somatic-interface-stack.v1";

/// Broad environment class relevant to sealing/hygiene architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SomaticContactEnvironment {
    Dry,
    Wet,
}

/// Semantic function of one layer, ordered from human-facing to structural.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SomaticLayerRole {
    ReplaceableContactLiner,
    CompliantSurface,
    TactileSensorSkin,
    VariableCompliance,
    FluidBarrier,
    ActuatorIsolation,
    StructuralBacking,
}

/// Sensor evidence the profile requires before later qualification may rely on it.
///
/// This enum states a requirement only. It does not claim that any installed
/// sensor measures the modality accurately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SomaticSensorModality {
    NormalForce,
    ShearForce,
    ContactArea,
    SurfaceTemperature,
    Humidity,
    Deformation,
    Slip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SomaticInterfaceLayerV1 {
    layer_id: String,
    role: SomaticLayerRole,
}

impl SomaticInterfaceLayerV1 {
    pub fn new(
        layer_id: impl Into<String>,
        role: SomaticLayerRole,
    ) -> Result<Self, SomaticInterfaceError> {
        let layer_id = layer_id.into().trim().to_owned();
        if layer_id.is_empty() || layer_id.len() > 128 {
            return Err(SomaticInterfaceError::InvalidLayerId);
        }
        Ok(Self { layer_id, role })
    }

    pub fn layer_id(&self) -> &str {
        &self.layer_id
    }

    pub const fn role(&self) -> SomaticLayerRole {
        self.role
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SomaticInterfaceError {
    InvalidProfileId,
    InvalidContactSiteId,
    EmptyLayerStack,
    TooManyLayers,
    InvalidLayerId,
    DuplicateLayerId,
    InvalidHumanFacingLayer,
    MissingStructuralBacking,
    MultipleStructuralBackings,
    StructuralBackingNotInnermost,
    MissingSensorRequirements,
    WetInterfaceRequiresReplaceableLiner,
    WetInterfaceRequiresFluidBarrier,
    FluidBarrierTooDeep,
}

/// Semantic architecture of one human-facing soft interface.
///
/// There is intentionally no `Default`: every profile must explicitly state its
/// layer stack, environment class, and required sensor evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SomaticInterfaceStackV1 {
    profile_id: String,
    contact_site_id: String,
    environment: SomaticContactEnvironment,
    layers_human_to_structure: Vec<SomaticInterfaceLayerV1>,
    required_sensor_modalities: BTreeSet<SomaticSensorModality>,
}

impl SomaticInterfaceStackV1 {
    pub fn new(
        profile_id: impl Into<String>,
        contact_site_id: impl Into<String>,
        environment: SomaticContactEnvironment,
        layers_human_to_structure: Vec<SomaticInterfaceLayerV1>,
        required_sensor_modalities: impl IntoIterator<Item = SomaticSensorModality>,
    ) -> Result<Self, SomaticInterfaceError> {
        let profile_id = profile_id.into().trim().to_owned();
        let contact_site_id = contact_site_id.into().trim().to_owned();
        if profile_id.is_empty() || profile_id.len() > 160 {
            return Err(SomaticInterfaceError::InvalidProfileId);
        }
        if contact_site_id.is_empty() || contact_site_id.len() > 128 {
            return Err(SomaticInterfaceError::InvalidContactSiteId);
        }
        if layers_human_to_structure.is_empty() {
            return Err(SomaticInterfaceError::EmptyLayerStack);
        }
        if layers_human_to_structure.len() > 32 {
            return Err(SomaticInterfaceError::TooManyLayers);
        }

        let mut layer_ids = BTreeSet::new();
        for layer in &layers_human_to_structure {
            if !layer_ids.insert(layer.layer_id.clone()) {
                return Err(SomaticInterfaceError::DuplicateLayerId);
            }
        }

        if !matches!(
            layers_human_to_structure[0].role,
            SomaticLayerRole::ReplaceableContactLiner | SomaticLayerRole::CompliantSurface
        ) {
            return Err(SomaticInterfaceError::InvalidHumanFacingLayer);
        }

        let structural_indices = layers_human_to_structure
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                (layer.role == SomaticLayerRole::StructuralBacking).then_some(index)
            })
            .collect::<Vec<_>>();
        if structural_indices.is_empty() {
            return Err(SomaticInterfaceError::MissingStructuralBacking);
        }
        if structural_indices.len() != 1 {
            return Err(SomaticInterfaceError::MultipleStructuralBackings);
        }
        if structural_indices[0] != layers_human_to_structure.len() - 1 {
            return Err(SomaticInterfaceError::StructuralBackingNotInnermost);
        }

        let required_sensor_modalities = required_sensor_modalities.into_iter().collect();
        if required_sensor_modalities.is_empty() {
            return Err(SomaticInterfaceError::MissingSensorRequirements);
        }

        if environment == SomaticContactEnvironment::Wet {
            if layers_human_to_structure[0].role != SomaticLayerRole::ReplaceableContactLiner {
                return Err(SomaticInterfaceError::WetInterfaceRequiresReplaceableLiner);
            }
            let barrier_index = layers_human_to_structure
                .iter()
                .position(|layer| layer.role == SomaticLayerRole::FluidBarrier)
                .ok_or(SomaticInterfaceError::WetInterfaceRequiresFluidBarrier)?;
            let first_protected_inner_index = layers_human_to_structure
                .iter()
                .position(|layer| {
                    matches!(
                        layer.role,
                        SomaticLayerRole::ActuatorIsolation | SomaticLayerRole::StructuralBacking
                    )
                })
                .unwrap_or(structural_indices[0]);
            if barrier_index >= first_protected_inner_index {
                return Err(SomaticInterfaceError::FluidBarrierTooDeep);
            }
        }

        Ok(Self {
            profile_id,
            contact_site_id,
            environment,
            layers_human_to_structure,
            required_sensor_modalities,
        })
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn contact_site_id(&self) -> &str {
        &self.contact_site_id
    }

    pub const fn environment(&self) -> SomaticContactEnvironment {
        self.environment
    }

    pub fn layers_human_to_structure(&self) -> &[SomaticInterfaceLayerV1] {
        &self.layers_human_to_structure
    }

    pub fn required_sensor_modalities(&self) -> &BTreeSet<SomaticSensorModality> {
        &self.required_sensor_modalities
    }

    pub fn requires_sensor(&self, modality: SomaticSensorModality) -> bool {
        self.required_sensor_modalities.contains(&modality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(id: &str, role: SomaticLayerRole) -> SomaticInterfaceLayerV1 {
        SomaticInterfaceLayerV1::new(id, role).unwrap()
    }

    fn core_sensors() -> [SomaticSensorModality; 3] {
        [
            SomaticSensorModality::NormalForce,
            SomaticSensorModality::ShearForce,
            SomaticSensorModality::SurfaceTemperature,
        ]
    }

    #[test]
    fn dry_stack_can_express_compliant_sensorized_interface() {
        let stack = SomaticInterfaceStackV1::new(
            "somatic.hand.v1",
            "right_hand_surface",
            SomaticContactEnvironment::Dry,
            vec![
                layer("surface", SomaticLayerRole::CompliantSurface),
                layer("skin", SomaticLayerRole::TactileSensorSkin),
                layer("compliance", SomaticLayerRole::VariableCompliance),
                layer("isolation", SomaticLayerRole::ActuatorIsolation),
                layer("frame", SomaticLayerRole::StructuralBacking),
            ],
            core_sensors(),
        )
        .unwrap();
        assert!(stack.requires_sensor(SomaticSensorModality::NormalForce));
        assert_eq!(stack.layers_human_to_structure().len(), 5);
    }

    #[test]
    fn wet_stack_requires_replaceable_liner_and_barrier() {
        let no_liner = SomaticInterfaceStackV1::new(
            "wet.v1",
            "contact_surface",
            SomaticContactEnvironment::Wet,
            vec![
                layer("surface", SomaticLayerRole::CompliantSurface),
                layer("barrier", SomaticLayerRole::FluidBarrier),
                layer("frame", SomaticLayerRole::StructuralBacking),
            ],
            core_sensors(),
        );
        assert_eq!(
            no_liner,
            Err(SomaticInterfaceError::WetInterfaceRequiresReplaceableLiner)
        );

        let no_barrier = SomaticInterfaceStackV1::new(
            "wet.v1",
            "contact_surface",
            SomaticContactEnvironment::Wet,
            vec![
                layer("liner", SomaticLayerRole::ReplaceableContactLiner),
                layer("surface", SomaticLayerRole::CompliantSurface),
                layer("frame", SomaticLayerRole::StructuralBacking),
            ],
            core_sensors(),
        );
        assert_eq!(
            no_barrier,
            Err(SomaticInterfaceError::WetInterfaceRequiresFluidBarrier)
        );
    }

    #[test]
    fn wet_barrier_must_protect_inner_actuation_structure() {
        let stack = SomaticInterfaceStackV1::new(
            "wet.v1",
            "contact_surface",
            SomaticContactEnvironment::Wet,
            vec![
                layer("liner", SomaticLayerRole::ReplaceableContactLiner),
                layer("surface", SomaticLayerRole::CompliantSurface),
                layer("isolation", SomaticLayerRole::ActuatorIsolation),
                layer("barrier", SomaticLayerRole::FluidBarrier),
                layer("frame", SomaticLayerRole::StructuralBacking),
            ],
            core_sensors(),
        );
        assert_eq!(stack, Err(SomaticInterfaceError::FluidBarrierTooDeep));
    }

    #[test]
    fn structural_backing_is_unique_and_innermost() {
        let missing = SomaticInterfaceStackV1::new(
            "dry.v1",
            "surface",
            SomaticContactEnvironment::Dry,
            vec![layer("surface", SomaticLayerRole::CompliantSurface)],
            core_sensors(),
        );
        assert_eq!(missing, Err(SomaticInterfaceError::MissingStructuralBacking));

        let not_last = SomaticInterfaceStackV1::new(
            "dry.v1",
            "surface",
            SomaticContactEnvironment::Dry,
            vec![
                layer("surface", SomaticLayerRole::CompliantSurface),
                layer("frame", SomaticLayerRole::StructuralBacking),
                layer("sensor", SomaticLayerRole::TactileSensorSkin),
            ],
            core_sensors(),
        );
        assert_eq!(
            not_last,
            Err(SomaticInterfaceError::StructuralBackingNotInnermost)
        );
    }

    #[test]
    fn duplicate_layer_identity_fails_closed() {
        let stack = SomaticInterfaceStackV1::new(
            "dry.v1",
            "surface",
            SomaticContactEnvironment::Dry,
            vec![
                layer("same", SomaticLayerRole::CompliantSurface),
                layer("same", SomaticLayerRole::StructuralBacking),
            ],
            core_sensors(),
        );
        assert_eq!(stack, Err(SomaticInterfaceError::DuplicateLayerId));
    }

    #[test]
    fn sensor_requirements_must_be_explicit() {
        let stack = SomaticInterfaceStackV1::new(
            "dry.v1",
            "surface",
            SomaticContactEnvironment::Dry,
            vec![
                layer("surface", SomaticLayerRole::CompliantSurface),
                layer("frame", SomaticLayerRole::StructuralBacking),
            ],
            [],
        );
        assert_eq!(stack, Err(SomaticInterfaceError::MissingSensorRequirements));
    }
}
