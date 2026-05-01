// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

/// Physical transport class for radio tier routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainTransportClass {
    /// Local mesh (LoRa, B.A.T.M.A.N., <1km)
    LocalMesh,
    /// Metro relay (Yggdrasil, Iroh, <50km)
    MetroRelay,
    /// Regional relay (satellite uplink, <5000km)
    RegionalRelay,
    /// Interplanetary relay (DTN, light-minutes)
    InterplanetaryRelay,
}

/// Transport capabilities and preferences for a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainTransportProfile {
    /// Whether the domain tolerates store-and-forward fragmentation.
    pub store_and_forward_required: bool,
    /// Ordered transport preferences (most preferred first).
    preferred_transports: Vec<DomainTransportClass>,
}

impl Default for DomainTransportProfile {
    fn default() -> Self {
        Self {
            store_and_forward_required: false,
            preferred_transports: vec![
                DomainTransportClass::LocalMesh,
                DomainTransportClass::MetroRelay,
                DomainTransportClass::RegionalRelay,
            ],
        }
    }
}

impl DomainTransportProfile {
    /// Return transport classes in priority order.
    pub fn priority_order(&self) -> Vec<DomainTransportClass> {
        self.preferred_transports.clone()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DomainProfile {
    pub primary_domain: String,
    pub kind: String,
    pub capabilities: Vec<String>,
    /// Transport class preferences for radio tier routing.
    #[serde(default)]
    pub transport: DomainTransportProfile,
}

impl DomainProfile {
    pub fn underwater() -> Self {
        Self {
            primary_domain: "underwater".into(),
            kind: "underwater".into(),
            capabilities: vec!["sonar".into()],
            transport: DomainTransportProfile {
                store_and_forward_required: true,
                ..Default::default()
            },
        }
    }
    pub fn subterranean() -> Self {
        Self {
            primary_domain: "subterranean".into(),
            kind: "subterranean".into(),
            capabilities: vec!["lidar".into()],
            ..Default::default()
        }
    }
    pub fn deep_space() -> Self {
        Self {
            primary_domain: "deep_space".into(),
            kind: "deep_space".into(),
            capabilities: vec!["star_tracker".into()],
            transport: DomainTransportProfile {
                store_and_forward_required: true,
                preferred_transports: vec![
                    DomainTransportClass::InterplanetaryRelay,
                    DomainTransportClass::RegionalRelay,
                ],
            },
        }
    }

    /// Check if this domain supports the given transport class.
    pub fn supports_transport(&self, class: DomainTransportClass) -> bool {
        self.transport.preferred_transports.contains(&class)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlatformCapabilityProfile {
    pub manipulation: f64,
    pub locomotion: f64,
    pub perception: f64,
    pub communication: f64,
}

impl PlatformCapabilityProfile {
    pub fn for_platform(platform: symthaea_core::embodiment::EmbodimentPlatform) -> Self {
        use symthaea_core::embodiment::EmbodimentPlatform;
        match platform {
            EmbodimentPlatform::Humanoid => Self {
                manipulation: 0.8,
                locomotion: 0.9,
                perception: 0.7,
                communication: 0.5,
            },
            EmbodimentPlatform::Quadrotor => Self {
                manipulation: 0.0,
                locomotion: 0.9,
                perception: 0.8,
                communication: 0.7,
            },
            EmbodimentPlatform::Vehicle => Self {
                manipulation: 0.0,
                locomotion: 1.0,
                perception: 0.6,
                communication: 0.8,
            },
            EmbodimentPlatform::Helicopter => Self {
                manipulation: 0.0,
                locomotion: 0.95,
                perception: 0.7,
                communication: 0.6,
            },
            EmbodimentPlatform::Auv => Self {
                manipulation: 0.3,
                locomotion: 0.7,
                perception: 0.5,
                communication: 0.2,
            },
            EmbodimentPlatform::Manipulator => Self {
                manipulation: 1.0,
                locomotion: 0.0,
                perception: 0.6,
                communication: 0.4,
            },
            EmbodimentPlatform::Exoskeleton => Self {
                manipulation: 0.5,
                locomotion: 0.8,
                perception: 0.3,
                communication: 0.3,
            },
            EmbodimentPlatform::Surgical => Self {
                manipulation: 1.0,
                locomotion: 0.0,
                perception: 0.9,
                communication: 0.5,
            },
            EmbodimentPlatform::Orbital => Self {
                manipulation: 0.9,
                locomotion: 0.0,
                perception: 0.4,
                communication: 0.3,
            },
            EmbodimentPlatform::Quadruped => Self {
                manipulation: 0.0,
                locomotion: 0.85,
                perception: 0.7,
                communication: 0.5,
            },
            _ => Self::default(),
        }
    }
    pub fn supports_domain(&self, _domain: &str) -> bool {
        true
    }
    pub fn preferred_domain_profile(&self) -> DomainProfile {
        DomainProfile::default()
    }
}
