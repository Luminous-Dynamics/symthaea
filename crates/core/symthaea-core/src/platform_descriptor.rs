// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, capability-composed descriptors for embodied platforms.
//!
//! `EmbodimentPlatform` remains the stable preset identity used by existing
//! platform plugins. `PlatformDescriptor` describes what a body is and can do
//! without granting permission to do it. Runtime authority remains a separate
//! concern enforced by the existing safety/authorization paths.

use serde::{Deserialize, Serialize};

use crate::embodiment::EmbodimentPlatform;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DescriptorEvidence {
    /// Built-in qualitative preset; useful for routing, not qualification.
    PresetHeuristic,
    /// Declared by a platform/plugin but not independently measured.
    Declared,
    /// Backed by measured platform evidence.
    Measured,
    /// Backed by a platform-specific qualification campaign.
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
    pub preset: EmbodimentPlatform,
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

impl EmbodimentPlatform {
    /// Return the canonical descriptive preset for this platform identity.
    ///
    /// Presets are deliberately marked `PresetHeuristic`: they are sufficient
    /// for capability routing and discovery, but are not hardware qualification
    /// evidence and never grant actuator authority.
    pub fn descriptor(self) -> PlatformDescriptor {
        use AutonomyClass::*;
        use CommunicationCapability::*;
        use DescriptorEvidence::PresetHeuristic;
        use EnergyClass::*;
        use ManipulationCapability::*;
        use MissionRole::*;
        use MobilityClass::*;
        use MorphologyClass::*;
        use OperatingEnvironment::*;
        use SafetyCriticality::*;
        use SensorCapability::*;

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
            preset: self,
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
            evidence: PresetHeuristic,
        };

        match self {
            EmbodimentPlatform::None => d(
                "none", None, vec![MobilityClass::None], vec![], vec![], vec![], vec![Unknown],
                vec![], vec![], Low, Advisory, CapabilityStrengths::ZERO,
            ),
            EmbodimentPlatform::Humanoid => d(
                "humanoid", Humanoid, vec![Legged], vec![TerrestrialSurface, HabitatInterior],
                vec![GeneralManipulation, ToolUse, MaterialHandling],
                vec![Vision, Depth, Inertial, ForceTorque, Proprioception],
                vec![Battery], vec![LocalWireless, Mesh],
                vec![Mobility, Manipulation, Inspection, Logistics, Servicing],
                LifeCritical, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.8, locomotion: 0.9, perception: 0.7, communication: 0.5 },
            ),
            EmbodimentPlatform::Quadrotor => d(
                "quadrotor", Rotorcraft, vec![MobilityClass::Rotorcraft], vec![Aerial], vec![],
                vec![Vision, Depth, Inertial, Position], vec![Battery], vec![LocalWireless, Mesh],
                vec![Mobility, Inspection, Science], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.9, perception: 0.8, communication: 0.7 },
            ),
            EmbodimentPlatform::Vehicle => d(
                "vehicle", GroundVehicle, vec![Wheeled], vec![TerrestrialSurface], vec![],
                vec![Vision, Depth, Inertial, Position], vec![Battery, Fuel, Hybrid], vec![LocalWireless, Mesh, LongRange],
                vec![Mobility, Transport, Logistics], LifeCritical, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.0, locomotion: 1.0, perception: 0.6, communication: 0.8 },
            ),
            EmbodimentPlatform::Helicopter => d(
                "helicopter", Rotorcraft, vec![MobilityClass::Rotorcraft], vec![Aerial], vec![],
                vec![Vision, Inertial, Position], vec![Fuel, Hybrid], vec![LongRange],
                vec![Mobility, Transport, Inspection], LifeCritical, Supervised,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.95, perception: 0.7, communication: 0.6 },
            ),
            EmbodimentPlatform::Auv => d(
                "auv", MarineVehicle, vec![Swimming, Buoyant], vec![Underwater],
                vec![GeneralManipulation], vec![Sonar, Inertial, Position, Environmental],
                vec![Battery], vec![StoreAndForward, LongRange],
                vec![Mobility, Inspection, Science, Servicing], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.3, locomotion: 0.7, perception: 0.5, communication: 0.2 },
            ),
            EmbodimentPlatform::Manipulator => d(
                "manipulator", Manipulator, vec![MobilityClass::None], vec![HabitatInterior, GeneralInfrastructure],
                vec![GeneralManipulation, PrecisionManipulation, ToolUse, MaterialHandling],
                vec![ForceTorque, Proprioception, Vision], vec![ExternalGrid], vec![DirectAttached, LocalWireless],
                vec![Manipulation, Servicing, Logistics], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 1.0, locomotion: 0.0, perception: 0.6, communication: 0.4 },
            ),
            EmbodimentPlatform::Exoskeleton => d(
                "exoskeleton", Wearable, vec![HumanCoupled], vec![TerrestrialSurface, HabitatInterior],
                vec![HumanAssistance], vec![ForceTorque, Proprioception, Inertial], vec![HumanCoupled, Battery],
                vec![DirectAttached, LocalWireless], vec![Mobility, Manipulation, Care], LifeCritical, HumanCoupled,
                CapabilityStrengths { manipulation: 0.5, locomotion: 0.8, perception: 0.3, communication: 0.3 },
            ),
            EmbodimentPlatform::Surgical => d(
                "surgical", SurgicalManipulator, vec![MobilityClass::None], vec![HabitatInterior],
                vec![PrecisionManipulation, SurgicalInteraction], vec![Vision, Depth, ForceTorque, Biomedical],
                vec![ExternalGrid], vec![DirectAttached], vec![Surgery, Care, Manipulation], LifeCritical, Supervised,
                CapabilityStrengths { manipulation: 1.0, locomotion: 0.0, perception: 0.9, communication: 0.5 },
            ),
            EmbodimentPlatform::Orbital => d(
                "orbital-servicer", OrbitalServicer, vec![OrbitalFreeFlight], vec![OrbitalMicrogravity],
                vec![GeneralManipulation, PrecisionManipulation, ServicingToolUse],
                vec![Vision, Inertial, ForceTorque, OrbitalNavigation], vec![Battery, Solar],
                vec![LongRange, StoreAndForward, Interplanetary], vec![Mobility, Manipulation, Servicing, Inspection],
                High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.9, locomotion: 0.0, perception: 0.4, communication: 0.3 },
            ),
            EmbodimentPlatform::Quadruped => d(
                "quadruped", Quadruped, vec![Legged], vec![TerrestrialSurface], vec![],
                vec![Vision, Depth, Inertial, Proprioception], vec![Battery], vec![LocalWireless, Mesh],
                vec![Mobility, Inspection, Logistics], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.85, perception: 0.7, communication: 0.5 },
            ),
            EmbodimentPlatform::Subterranean => d(
                "subterranean", ProcessMachine, vec![Subterranean, Tracked], vec![OperatingEnvironment::Subterranean],
                vec![Excavation, MaterialHandling, ToolUse], vec![Depth, Inertial, Environmental, ForceTorque],
                vec![Battery, ExternalGrid], vec![Mesh, StoreAndForward], vec![Mobility, Mining, Construction, Inspection],
                High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.7, locomotion: 0.6, perception: 0.6, communication: 0.3 },
            ),
            EmbodimentPlatform::Infrastructure => d(
                "infrastructure", FixedInstallation, vec![MobilityClass::None], vec![GeneralInfrastructure],
                vec![ProcessControl], vec![Environmental, NetworkState], vec![ExternalGrid, Solar, Hybrid],
                vec![DirectAttached, Mesh, LongRange], vec![HabitatOperations, Communications, Computing],
                High, Cooperative,
                CapabilityStrengths { manipulation: 0.2, locomotion: 0.0, perception: 0.7, communication: 1.0 },
            ),
            EmbodimentPlatform::Scavenger => d(
                "scavenger", ProcessMachine, vec![Wheeled, Tracked], vec![TerrestrialSurface, GeneralInfrastructure],
                vec![MaterialHandling, ToolUse, GeneralManipulation], vec![Vision, Depth, ForceTorque], vec![Battery],
                vec![Mesh, LocalWireless], vec![Recycling, Logistics, Servicing], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.8, locomotion: 0.5, perception: 0.7, communication: 0.5 },
            ),
            EmbodimentPlatform::Agribot => d(
                "agribot", GroundVehicle, vec![Wheeled], vec![Agricultural, TerrestrialSurface],
                vec![GeneralManipulation, ToolUse, MaterialHandling], vec![Vision, Depth, Environmental, Position],
                vec![Battery, Solar], vec![Mesh, LocalWireless], vec![Agriculture, Inspection, Logistics],
                Moderate, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.6, locomotion: 0.6, perception: 0.8, communication: 0.5 },
            ),
            EmbodimentPlatform::Biota => d(
                "biota", GroundVehicle, vec![Wheeled], vec![AnimalProximate, TerrestrialSurface],
                vec![], vec![Vision, Depth, Environmental], vec![Battery], vec![Mesh, LocalWireless],
                vec![EcologicalStewardship, Inspection], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.1, locomotion: 0.5, perception: 0.9, communication: 0.5 },
            ),
            EmbodimentPlatform::Clime => d(
                "clime", FixedInstallation, vec![MobilityClass::None], vec![HabitatInterior, GeneralInfrastructure],
                vec![ProcessControl], vec![Environmental, NetworkState], vec![ExternalGrid], vec![DirectAttached, Mesh],
                vec![HabitatOperations, Care], LifeCritical, Cooperative,
                CapabilityStrengths { manipulation: 0.3, locomotion: 0.0, perception: 0.9, communication: 0.6 },
            ),
            EmbodimentPlatform::CareProvider => d(
                "care-provider", Virtual, vec![MobilityClass::Virtual], vec![Digital], vec![],
                vec![NetworkState], vec![Virtual], vec![DirectAttached, LongRange], vec![Care, Computing],
                High, VirtualAgent,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.0, perception: 0.6, communication: 1.0 },
            ),
            EmbodimentPlatform::Browser => d(
                "browser", Virtual, vec![MobilityClass::Virtual], vec![Digital], vec![], vec![NetworkState], vec![Virtual],
                vec![DirectAttached, LongRange], vec![Inspection, Computing, Communications], Moderate, VirtualAgent,
                CapabilityStrengths { manipulation: 0.0, locomotion: 0.0, perception: 0.8, communication: 1.0 },
            ),
            EmbodimentPlatform::Phone => d(
                "phone", Virtual, vec![MobilityClass::Virtual], vec![Digital], vec![], vec![Vision, NetworkState], vec![Virtual],
                vec![DirectAttached, LocalWireless, LongRange], vec![Inspection, Computing, Communications], Moderate, Supervised,
                CapabilityStrengths { manipulation: 0.1, locomotion: 0.0, perception: 0.8, communication: 1.0 },
            ),
            EmbodimentPlatform::Desktop => d(
                "desktop", Virtual, vec![MobilityClass::Virtual], vec![Digital], vec![], vec![Vision, NetworkState], vec![Virtual],
                vec![DirectAttached, LongRange], vec![Inspection, Computing, Communications], Moderate, Supervised,
                CapabilityStrengths { manipulation: 0.1, locomotion: 0.0, perception: 0.8, communication: 1.0 },
            ),
            EmbodimentPlatform::Detritivore => d(
                "detritivore", ProcessMachine, vec![MobilityClass::None], vec![GeneralInfrastructure],
                vec![MaterialHandling, ProcessControl], vec![ForceTorque, Environmental], vec![ExternalGrid],
                vec![DirectAttached, Mesh], vec![Recycling, Servicing], High, BoundedAutonomy,
                CapabilityStrengths { manipulation: 0.7, locomotion: 0.0, perception: 0.5, communication: 0.4 },
            ),
        }
    }
}

// Internal alias used only to keep the Orbital preset readable without adding
// a separate public capability for every possible tool family.
const ServicingToolUse: ManipulationCapability = ManipulationCapability::ToolUse;

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_PLATFORMS: [EmbodimentPlatform; 21] = [
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
        // Detritivore intentionally tested separately below because changing
        // the stable enum roster should force this census to be reviewed.
    ];

    #[test]
    fn every_preset_descriptor_is_structurally_valid() {
        for platform in ALL_PLATFORMS {
            let descriptor = platform.descriptor();
            assert_eq!(descriptor.preset, platform);
            assert!(descriptor.is_structurally_valid(), "invalid {platform:?}");
            assert_eq!(descriptor.evidence, DescriptorEvidence::PresetHeuristic);
        }
        assert!(EmbodimentPlatform::Detritivore.descriptor().is_structurally_valid());
    }

    #[test]
    fn capability_is_not_authority() {
        let descriptor = EmbodimentPlatform::Manipulator.descriptor();
        assert!(descriptor.supports_role(MissionRole::Manipulation));
        // Compile-time/type-level boundary: the descriptor contains no command
        // or authorization object. This regression test pins its evidence state
        // as heuristic so discovery cannot masquerade as qualification.
        assert_eq!(descriptor.evidence, DescriptorEvidence::PresetHeuristic);
    }

    #[test]
    fn deep_space_platform_advertises_store_and_forward() {
        let descriptor = EmbodimentPlatform::Orbital.descriptor();
        assert!(descriptor.supports_environment(OperatingEnvironment::OrbitalMicrogravity));
        assert!(descriptor.communications.contains(&CommunicationCapability::StoreAndForward));
    }
}
