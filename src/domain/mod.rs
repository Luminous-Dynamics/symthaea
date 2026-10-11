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
                // Water attenuates local RF mesh (LoRa/B.A.T.M.A.N.) far too
                // much to be usable -- omit it entirely rather than just
                // deprioritizing it, so `supports_transport` correctly
                // refuses it too.
                preferred_transports: vec![
                    DomainTransportClass::MetroRelay,
                    DomainTransportClass::RegionalRelay,
                ],
            },
        }
    }
    pub fn subterranean() -> Self {
        Self {
            primary_domain: "subterranean".into(),
            kind: "subterranean".into(),
            capabilities: vec!["lidar".into()],
            transport: DomainTransportProfile {
                // Underground comms are delay-tolerant / relay-dependent --
                // allow store-and-forward fragmentation rather than
                // blocking when no tier meets the nominal bandwidth
                // preference.
                store_and_forward_required: true,
                ..Default::default()
            },
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

/// Legacy four-axis projection of the canonical typed platform descriptor.
///
/// New code should prefer `symthaea_core::platform_descriptor::PlatformDescriptor`.
/// This compatibility surface remains so older domain-routing callers do not
/// need to migrate atomically.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlatformCapabilityProfile {
    pub manipulation: f64,
    pub locomotion: f64,
    pub perception: f64,
    pub communication: f64,
}

impl PlatformCapabilityProfile {
    pub fn for_platform(platform: symthaea_core::embodiment::EmbodimentPlatform) -> Self {
        let strengths = platform.descriptor().strengths;
        Self {
            manipulation: strengths.manipulation,
            locomotion: strengths.locomotion,
            perception: strengths.perception,
            communication: strengths.communication,
        }
    }

    pub fn supports_domain(&self, _domain: &str) -> bool {
        true
    }

    pub fn preferred_domain_profile(&self) -> DomainProfile {
        DomainProfile::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment::EmbodimentPlatform;
    use symthaea_core::platform_descriptor::{MissionRole, OperatingEnvironment};

    #[test]
    fn legacy_profile_is_projection_of_canonical_descriptor() {
        let platform = EmbodimentPlatform::Humanoid;
        let legacy = PlatformCapabilityProfile::for_platform(platform);
        let strengths = platform.descriptor().strengths;
        assert_eq!(legacy.manipulation, strengths.manipulation);
        assert_eq!(legacy.locomotion, strengths.locomotion);
        assert_eq!(legacy.perception, strengths.perception);
        assert_eq!(legacy.communication, strengths.communication);
    }

    #[test]
    fn newer_platforms_no_longer_collapse_to_zero_capability() {
        let profile = PlatformCapabilityProfile::for_platform(EmbodimentPlatform::Infrastructure);
        assert!(profile.perception > 0.0);
        assert!(profile.communication > 0.0);
    }

    #[test]
    fn typed_descriptor_carries_role_and_environment_separately() {
        let descriptor = EmbodimentPlatform::Orbital.descriptor();
        assert!(descriptor.supports_role(MissionRole::Servicing));
        assert!(descriptor.supports_environment(OperatingEnvironment::OrbitalMicrogravity));
    }
}
