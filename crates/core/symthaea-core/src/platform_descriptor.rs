// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, capability-composed descriptors for embodied platforms.
//!
//! `EmbodimentPlatform` remains the stable preset identity used by existing
//! platform plugins. `PlatformDescriptor` describes what a body is and can do
//! without granting permission to do it. Runtime authority remains a separate
//! concern enforced by the existing safety/authorization paths.

use serde::{Deserialize, Serialize};

use crate::embodiment::{EmbodimentPlatform, PlatformRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DescriptorEvidence {
    PresetHeuristic,
    Declared,
    Measured,
    Qualified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphologyClass {
    None,
    Virtual,
    FixedInstallation,
    Humanoid,
    Quadruped,
    Rotorcraft,
    GroundVehicle,
    MarineVehicle,
    Manipulator,
    Wearable,
    SurgicalManipulator,
    Spacecraft,
    OrbitalServicer,
    ProcessMachine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobilityClass {
    None,
    Virtual,
    Wheeled,
    Tracked,
    Legged,
    Rotorcraft,
    Buoyant,
    Swimming,
    OrbitalFreeFlight,
    Subterranean,
    HumanCoupled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatingEnvironment {
    TerrestrialSurface,
    PlanetarySurface,
    AirlessSurface,
    Aerial,
    Underwater,
    Subterranean,
    OrbitalMicrogravity,
    HabitatInterior,
    Agricultural,
    AnimalProximate,
    Digital,
    GeneralInfrastructure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ManipulationCapability {
    GeneralManipulation,
    PrecisionManipulation,
    HumanAssistance,
    SurgicalInteraction,
    Excavation,
    MaterialHandling,
    ToolUse,
    ProcessControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorCapability {
    Vision,
    Depth,
    Inertial,
    Position,
    ForceTorque,
    Proprioception,
    Sonar,
    Environmental,
    Biomedical,
    OrbitalNavigation,
    NetworkState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergyClass {
    Unknown,
    Virtual,
    HumanCoupled,
    Battery,
    ExternalGrid,
    Fuel,
    Solar,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommunicationCapability {
    DirectAttached,
    LocalWireless,
    Mesh,
    LongRange,
    StoreAndForward,
    Interplanetary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MissionRole {
    Mobility,
    Transport,
    Logistics,
    Inspection,
    Manipulation,
    Construction,
    Mining,
    Servicing,
    Science,
    Agriculture,
    HabitatOperations,
    Recycling,
    EcologicalStewardship,
    Care,
    Surgery,
    Communications,
    Navigation,
    Computing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyCriticality {
    Low,
    Moderate,
    High,
    LifeCritical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AutonomyClass {
    VirtualAgent,
    Advisory,
    Supervised,
    BoundedAutonomy,
    Cooperative,
    HumanCoupled,
}

/// Legacy four-axis capability projection retained for compatibility.
///
/// Values are normalized to [0, 1]. They are routing hints, not certified
/// performance limits. New code should prefer the typed descriptor fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityStrengths {
    pub manipulation: f64,
    pub locomotion: f64,
    pub perception: f64,
    pub communication: f64,
}

impl CapabilityStrengths {
    pub const ZERO: Self = Self {
        manipulation: 0.0,
        locomotion: 0.0,
        perception: 0.0,
        communication: 0.0,
    };

    pub fn is_valid(self) -> bool {
        [
            self.manipulation,
            self.locomotion,
            self.perception,
            self.communication,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

/// Domain-neutral description of a platform body and its declared capabilities.
///
/// This type is intentionally non-executable: it carries no motor command,
/// actuator handle, token, capability grant, or authorization state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlatformDescriptor {
    /// Existing named preset, when this descriptor came from one.
    pub preset: Option<EmbodimentPlatform>,
    pub name: String,
    pub morphology: MorphologyClass,
    pub mobility: Vec<MobilityClass>,
    pub environments: Vec<OperatingEnvironment>,
    pub manipulation: Vec<ManipulationCapability>,
    pub sensors: Vec<SensorCapability>,
    pub energy: Vec<EnergyClass>,
    pub communications: Vec<CommunicationCapability>,
    pub roles: Vec<MissionRole>,
    pub safety_criticality: SafetyCriticality,
    pub autonomy: AutonomyClass,
    pub strengths: CapabilityStrengths,
    pub evidence: DescriptorEvidence,
}

impl PlatformDescriptor {
    /// Start a descriptor for a body that does not need a named enum preset.
    /// Conservative defaults are intentional: supervised autonomy, high safety
    /// criticality, and zero capability strengths until explicitly declared.
    pub fn composed(name: impl Into<String>, morphology: MorphologyClass) -> Self {
        Self {
            preset: None,
            name: name.into(),
            morphology,
            mobility: Vec::new(),
            environments: Vec::new(),
            manipulation: Vec::new(),
            sensors: Vec::new(),
            energy: vec![EnergyClass::Unknown],
            communications: Vec::new(),
            roles: Vec::new(),
            safety_criticality: SafetyCriticality::High,
            autonomy: AutonomyClass::Supervised,
            strengths: CapabilityStrengths::ZERO,
            evidence: DescriptorEvidence::Declared,
        }
    }

    pub fn with_role(mut self, role: MissionRole) -> Self {
        push_unique(&mut self.roles, role);
        self
    }

    pub fn with_environment(mut self, environment: OperatingEnvironment) -> Self {
        push_unique(&mut self.environments, environment);
        self
    }

    pub fn with_mobility(mut self, mobility: MobilityClass) -> Self {
        push_unique(&mut self.mobility, mobility);
        self
    }

    pub fn with_sensor(mut self, sensor: SensorCapability) -> Self {
        push_unique(&mut self.sensors, sensor);
        self
    }

    pub fn with_manipulation(mut self, capability: ManipulationCapability) -> Self {
        push_unique(&mut self.manipulation, capability);
        self
    }

    pub fn with_communication(mut self, capability: CommunicationCapability) -> Self {
        push_unique(&mut self.communications, capability);
        self
    }

    pub fn with_energy(mut self, energy: EnergyClass) -> Self {
        if self.energy == [EnergyClass::Unknown] {
            self.energy.clear();
        }
        push_unique(&mut self.energy, energy);
        self
    }

    pub fn with_strengths(mut self, strengths: CapabilityStrengths) -> Self {
        self.strengths = strengths;
        self
    }

    pub fn with_autonomy(mut self, autonomy: AutonomyClass) -> Self {
        self.autonomy = autonomy;
        self
    }

    pub fn with_safety_criticality(mut self, criticality: SafetyCriticality) -> Self {
        self.safety_criticality = criticality;
        self
    }

    pub fn with_evidence(mut self, evidence: DescriptorEvidence) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn supports_role(&self, role: MissionRole) -> bool {
        self.roles.contains(&role)
    }

    pub fn supports_environment(&self, environment: OperatingEnvironment) -> bool {
        self.environments.contains(&environment)
    }

    pub fn supports_mobility(&self, mobility: MobilityClass) -> bool {
        self.mobility.contains(&mobility)
    }

    pub fn is_structurally_valid(&self) -> bool {
        !self.name.trim().is_empty() && self.strengths.is_valid()
    }
}

fn push_unique<T: PartialEq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

/// Read-only capability discovery over the existing plugin registry.
///
/// This deliberately returns platform identities/descriptors only. It never
/// constructs a bridge or creates runtime authority.
pub trait PlatformRegistryDescriptorExt {
    fn registered_descriptors(&self) -> Vec<PlatformDescriptor>;
    fn platforms_supporting_role(&self, role: MissionRole) -> Vec<EmbodimentPlatform>;
    fn platforms_supporting(
        &self,
        role: MissionRole,
        environment: OperatingEnvironment,
    ) -> Vec<EmbodimentPlatform>;
}

impl PlatformRegistryDescriptorExt for PlatformRegistry {
    fn registered_descriptors(&self) -> Vec<PlatformDescriptor> {
        self.registered_platforms()
            .into_iter()
            .map(EmbodimentPlatform::descriptor)
            .collect()
    }

    fn platforms_supporting_role(&self, role: MissionRole) -> Vec<EmbodimentPlatform> {
        self.registered_platforms()
            .into_iter()
            .filter(|platform| platform.descriptor().supports_role(role))
            .collect()
    }

    fn platforms_supporting(
        &self,
        role: MissionRole,
        environment: OperatingEnvironment,
    ) -> Vec<EmbodimentPlatform> {
        self.registered_platforms()
            .into_iter()
            .filter(|platform| {
                let descriptor = platform.descriptor();
                descriptor.supports_role(role) && descriptor.supports_environment(environment)
            })
            .collect()
    }
}

impl EmbodimentPlatform {
    /// Return the canonical descriptive preset for this platform identity.
    ///
    /// Presets are deliberately marked `PresetHeuristic`: they are sufficient
    /// for capability routing and discovery, but are not hardware qualification
    /// evidence and never grant actuator authority.
    pub fn descriptor(self) -> PlatformDescriptor {
        use AutonomyClass as A;
        use CommunicationCapability as C;
        use EnergyClass as En;
        use ManipulationCapability as M;
        use MissionRole as R;
        use MobilityClass as Mob;
        use MorphologyClass as Morph;
        use OperatingEnvironment as Env;
        use SafetyCriticality as S;
        use SensorCapability as Se;

        let d = |name: &str,
                 morphology,
                 mobility: Vec<MobilityClass>,
                 environments: Vec<OperatingEnvironment>,
                 manipulation: Vec<ManipulationCapability>,
                 sensors: Vec<SensorCapability>,
                 energy: Vec<EnergyClass>,
                 communications: Vec<CommunicationCapability>,
                 roles: Vec<MissionRole>,
                 safety_criticality,
                 autonomy,
                 strengths| PlatformDescriptor {
            preset: Some(self),
            name: name.into(),
            morphology,
            mobility,
            environments,
            manipulation,
            sensors,
            energy,
            communications,
            roles,
            safety_criticality,
            autonomy,
            strengths,
            evidence: DescriptorEvidence::PresetHeuristic,
        };

        match self {
            EmbodimentPlatform::None => d(
                "none", Morph::None, vec![Mob::None], vec![], vec![], vec![], vec![En::Unknown],
                vec![], vec![], S::Low, A::Advisory, CapabilityStrengths::ZERO,
            ),
            EmbodimentPlatform::Humanoid => d(
                "humanoid", Morph::Humanoid, vec![Mob::Legged], vec![Env::TerrestrialSurface, Env::HabitatInterior],
                vec![M::GeneralManipulation, M::ToolUse, M::MaterialHandling],
                vec![Se::Vision, Se::Depth, Se::Inertial, Se::ForceTorque, Se::Proprioception],
                vec![En::Battery], vec![C::LocalWireless, C::Mesh],
                vec![R::Mobility, R::Manipulation, R::Inspection, R::Logistics, R::Servicing],
                S::LifeCritical, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.8, locomotion: 0.9, perception: 0.7, communication: 0.5 },
            ),
            EmbodimentPlatform::Quadrotor => d(
                "quadrotor", Morph::Rotorcraft, vec![Mob::Rotorcraft], vec![Env::Aerial], vec![],
                vec![Se::Vision, Se::Depth, Se::Inertial, Se::Position], vec![En::Battery], vec![C::LocalWireless, C::Mesh],
                vec![R::Mobility, R::Inspection, R::Science], S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.9, perception: 0.8, communication: 0.7 },
            ),
            EmbodimentPlatform::Vehicle => d(
                "vehicle", Morph::GroundVehicle, vec![Mob::Wheeled], vec![Env::TerrestrialSurface], vec![],
                vec![Se::Vision, Se::Depth, Se::Inertial, Se::Position], vec![En::Battery, En::Fuel, En::Hybrid],
                vec![C::LocalWireless, C::Mesh, C::LongRange], vec![R::Mobility, R::Transport, R::Logistics],
                S::LifeCritical, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.0, locomotion: 1.0, perception: 0.6, communication: 0.8 },
            ),
            EmbodimentPlatform::Helicopter => d(
                "helicopter", Morph::Rotorcraft, vec![Mob::Rotorcraft], vec![Env::Aerial], vec![],
                vec![Se::Vision, Se::Inertial, Se::Position], vec![En::Fuel, En::Hybrid], vec![C::LongRange],
                vec![R::Mobility, R::Transport, R::Inspection], S::LifeCritical, A::Supervised,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.95, perception: 0.7, communication: 0.6 },
            ),
            EmbodimentPlatform::Auv => d(
                "auv", Morph::MarineVehicle, vec![Mob::Swimming, Mob::Buoyant], vec![Env::Underwater],
                vec![M::GeneralManipulation], vec![Se::Sonar, Se::Inertial, Se::Position, Se::Environmental],
                vec![En::Battery], vec![C::StoreAndForward, C::LongRange],
                vec![R::Mobility, R::Inspection, R::Science, R::Servicing], S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.3, locomotion: 0.7, perception: 0.5, communication: 0.2 },
            ),
            EmbodimentPlatform::Manipulator => d(
                "manipulator", Morph::Manipulator, vec![Mob::None], vec![Env::HabitatInterior, Env::GeneralInfrastructure],
                vec![M::GeneralManipulation, M::PrecisionManipulation, M::ToolUse, M::MaterialHandling],
                vec![Se::ForceTorque, Se::Proprioception, Se::Vision], vec![En::ExternalGrid],
                vec![C::DirectAttached, C::LocalWireless], vec![R::Manipulation, R::Servicing, R::Logistics],
                S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 1.0, locomotion: 0.0, perception: 0.6, communication: 0.4 },
            ),
            EmbodimentPlatform::Exoskeleton => d(
                "exoskeleton", Morph::Wearable, vec![Mob::HumanCoupled], vec![Env::TerrestrialSurface, Env::HabitatInterior],
                vec![M::HumanAssistance], vec![Se::ForceTorque, Se::Proprioception, Se::Inertial],
                vec![En::HumanCoupled, En::Battery], vec![C::DirectAttached, C::LocalWireless],
                vec![R::Mobility, R::Manipulation, R::Care], S::LifeCritical, A::HumanCoupled,
                CapabilityStrengths { manipulation: 0.5, locomotion: 0.8, perception: 0.3, communication: 0.3 },
            ),
            EmbodimentPlatform::Surgical => d(
                "surgical", Morph::SurgicalManipulator, vec![Mob::None], vec![Env::HabitatInterior],
                vec![M::PrecisionManipulation, M::SurgicalInteraction], vec![Se::Vision, Se::Depth, Se::ForceTorque, Se::Biomedical],
                vec![En::ExternalGrid], vec![C::DirectAttached], vec![R::Surgery, R::Care, R::Manipulation],
                S::LifeCritical, A::Supervised,
                CapabilityStrengths { manipulation: 1.0, locomotion: 0.0, perception: 0.9, communication: 0.5 },
            ),
            EmbodimentPlatform::Orbital => d(
                "orbital-servicer", Morph::OrbitalServicer, vec![Mob::OrbitalFreeFlight], vec![Env::OrbitalMicrogravity],
                vec![M::GeneralManipulation, M::PrecisionManipulation, M::ToolUse],
                vec![Se::Vision, Se::Inertial, Se::ForceTorque, Se::OrbitalNavigation], vec![En::Battery, En::Solar],
                vec![C::LongRange, C::StoreAndForward, C::Interplanetary],
                vec![R::Mobility, R::Manipulation, R::Servicing, R::Inspection], S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.9, locomotion: 0.0, perception: 0.4, communication: 0.3 },
            ),
            EmbodimentPlatform::Quadruped => d(
                "quadruped", Morph::Quadruped, vec![Mob::Legged], vec![Env::TerrestrialSurface], vec![],
                vec![Se::Vision, Se::Depth, Se::Inertial, Se::Proprioception], vec![En::Battery], vec![C::LocalWireless, C::Mesh],
                vec![R::Mobility, R::Inspection, R::Logistics], S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.85, perception: 0.7, communication: 0.5 },
            ),
            EmbodimentPlatform::Subterranean => d(
                "subterranean", Morph::ProcessMachine, vec![Mob::Subterranean, Mob::Tracked], vec![Env::Subterranean],
                vec![M::Excavation, M::MaterialHandling, M::ToolUse], vec![Se::Depth, Se::Inertial, Se::Environmental, Se::ForceTorque],
                vec![En::Battery, En::ExternalGrid], vec![C::Mesh, C::StoreAndForward],
                vec![R::Mobility, R::Mining, R::Construction, R::Inspection], S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.7, locomotion: 0.6, perception: 0.6, communication: 0.3 },
            ),
            EmbodimentPlatform::Infrastructure => d(
                "infrastructure", Morph::FixedInstallation, vec![Mob::None], vec![Env::GeneralInfrastructure],
                vec![M::ProcessControl], vec![Se::Environmental, Se::NetworkState], vec![En::ExternalGrid, En::Solar, En::Hybrid],
                vec![C::DirectAttached, C::Mesh, C::LongRange], vec![R::HabitatOperations, R::Communications, R::Computing],
                S::High, A::Cooperative,
                CapabilityStrengths { manipulation: 0.2, locomotion: 0.0, perception: 0.7, communication: 1.0 },
            ),
            EmbodimentPlatform::Scavenger => d(
                "scavenger", Morph::ProcessMachine, vec![Mob::Wheeled, Mob::Tracked], vec![Env::TerrestrialSurface, Env::GeneralInfrastructure],
                vec![M::MaterialHandling, M::ToolUse, M::GeneralManipulation], vec![Se::Vision, Se::Depth, Se::ForceTorque],
                vec![En::Battery], vec![C::Mesh, C::LocalWireless], vec![R::Recycling, R::Logistics, R::Servicing],
                S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.8, locomotion: 0.5, perception: 0.7, communication: 0.5 },
            ),
            EmbodimentPlatform::Agribot => d(
                "agribot", Morph::GroundVehicle, vec![Mob::Wheeled], vec![Env::Agricultural, Env::TerrestrialSurface],
                vec![M::GeneralManipulation, M::ToolUse, M::MaterialHandling],
                vec![Se::Vision, Se::Depth, Se::Environmental, Se::Position], vec![En::Battery, En::Solar],
                vec![C::Mesh, C::LocalWireless], vec![R::Agriculture, R::Inspection, R::Logistics],
                S::Moderate, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.6, locomotion: 0.6, perception: 0.8, communication: 0.5 },
            ),
            EmbodimentPlatform::Biota => d(
                "biota", Morph::GroundVehicle, vec![Mob::Wheeled], vec![Env::AnimalProximate, Env::TerrestrialSurface],
                vec![], vec![Se::Vision, Se::Depth, Se::Environmental], vec![En::Battery], vec![C::Mesh, C::LocalWireless],
                vec![R::EcologicalStewardship, R::Inspection], S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.1, locomotion: 0.5, perception: 0.9, communication: 0.5 },
            ),
            EmbodimentPlatform::Clime => d(
                "clime", Morph::FixedInstallation, vec![Mob::None], vec![Env::HabitatInterior, Env::GeneralInfrastructure],
                vec![M::ProcessControl], vec![Se::Environmental, Se::NetworkState], vec![En::ExternalGrid],
                vec![C::DirectAttached, C::Mesh], vec![R::HabitatOperations, R::Care], S::LifeCritical, A::Cooperative,
                CapabilityStrengths { manipulation: 0.3, locomotion: 0.0, perception: 0.9, communication: 0.6 },
            ),
            EmbodimentPlatform::CareProvider => d(
                "care-provider", Morph::Virtual, vec![Mob::Virtual], vec![Env::Digital], vec![], vec![Se::NetworkState],
                vec![En::Virtual], vec![C::DirectAttached, C::LongRange], vec![R::Care, R::Computing], S::High, A::VirtualAgent,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.0, perception: 0.6, communication: 1.0 },
            ),
            EmbodimentPlatform::Browser => d(
                "browser", Morph::Virtual, vec![Mob::Virtual], vec![Env::Digital], vec![], vec![Se::NetworkState],
                vec![En::Virtual], vec![C::DirectAttached, C::LongRange], vec![R::Inspection, R::Computing, R::Communications],
                S::Moderate, A::VirtualAgent,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.0, perception: 0.8, communication: 1.0 },
            ),
            EmbodimentPlatform::Phone => d(
                "phone", Morph::Virtual, vec![Mob::Virtual], vec![Env::Digital], vec![], vec![Se::Vision, Se::NetworkState],
                vec![En::Virtual], vec![C::DirectAttached, C::LocalWireless, C::LongRange],
                vec![R::Inspection, R::Computing, R::Communications], S::Moderate, A::Supervised,
                CapabilityStrengths { manipulation: 0.1, locomotion: 0.0, perception: 0.8, communication: 1.0 },
            ),
            EmbodimentPlatform::Desktop => d(
                "desktop", Morph::Virtual, vec![Mob::Virtual], vec![Env::Digital], vec![], vec![Se::Vision, Se::NetworkState],
                vec![En::Virtual], vec![C::DirectAttached, C::LongRange],
                vec![R::Inspection, R::Computing, R::Communications], S::Moderate, A::Supervised,
                CapabilityStrengths { manipulation: 0.1, locomotion: 0.0, perception: 0.8, communication: 1.0 },
            ),
            EmbodimentPlatform::Detritivore => d(
                "detritivore", Morph::ProcessMachine, vec![Mob::None], vec![Env::GeneralInfrastructure],
                vec![M::MaterialHandling, M::ProcessControl], vec![Se::ForceTorque, Se::Environmental],
                vec![En::ExternalGrid], vec![C::DirectAttached, C::Mesh], vec![R::Recycling, R::Servicing],
                S::High, A::BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.7, locomotion: 0.0, perception: 0.5, communication: 0.4 },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_PLATFORMS: [EmbodimentPlatform; 22] = [
        EmbodimentPlatform::None,
        EmbodimentPlatform::Humanoid,
        EmbodimentPlatform::Quadrotor,
        EmbodimentPlatform::Vehicle,
        EmbodimentPlatform::Helicopter,
        EmbodimentPlatform::Auv,
        EmbodimentPlatform::Manipulator,
        EmbodimentPlatform::Exoskeleton,
        EmbodimentPlatform::Surgical,
        EmbodimentPlatform::Orbital,
        EmbodimentPlatform::Quadruped,
        EmbodimentPlatform::Subterranean,
        EmbodimentPlatform::Infrastructure,
        EmbodimentPlatform::Scavenger,
        EmbodimentPlatform::Agribot,
        EmbodimentPlatform::Biota,
        EmbodimentPlatform::Clime,
        EmbodimentPlatform::CareProvider,
        EmbodimentPlatform::Browser,
        EmbodimentPlatform::Phone,
        EmbodimentPlatform::Desktop,
        EmbodimentPlatform::Detritivore,
    ];

    #[test]
    fn every_preset_descriptor_is_structurally_valid() {
        for platform in ALL_PLATFORMS {
            let descriptor = platform.descriptor();
            assert_eq!(descriptor.preset, Some(platform));
            assert!(descriptor.is_structurally_valid(), "invalid {platform:?}");
            assert_eq!(descriptor.evidence, DescriptorEvidence::PresetHeuristic);
        }
    }

    #[test]
    fn composed_descriptor_needs_no_enum_variant() {
        let rover = PlatformDescriptor::composed("lunar-hauler", MorphologyClass::GroundVehicle)
            .with_environment(OperatingEnvironment::AirlessSurface)
            .with_environment(OperatingEnvironment::PlanetarySurface)
            .with_mobility(MobilityClass::Tracked)
            .with_role(MissionRole::Logistics)
            .with_role(MissionRole::Construction)
            .with_sensor(SensorCapability::Inertial)
            .with_sensor(SensorCapability::Depth)
            .with_energy(EnergyClass::Battery);
        assert_eq!(rover.preset, None);
        assert!(rover.supports_role(MissionRole::Construction));
        assert!(rover.supports_environment(OperatingEnvironment::AirlessSurface));
        assert_eq!(rover.evidence, DescriptorEvidence::Declared);
        assert!(rover.is_structurally_valid());
    }

    #[test]
    fn capability_is_not_authority_or_qualification() {
        let descriptor = EmbodimentPlatform::Manipulator.descriptor();
        assert!(descriptor.supports_role(MissionRole::Manipulation));
        assert_eq!(descriptor.evidence, DescriptorEvidence::PresetHeuristic);
    }

    #[test]
    fn orbital_preset_advertises_store_and_forward() {
        let descriptor = EmbodimentPlatform::Orbital.descriptor();
        assert!(descriptor.supports_environment(OperatingEnvironment::OrbitalMicrogravity));
        assert!(descriptor
            .communications
            .contains(&CommunicationCapability::StoreAndForward));
    }
}
