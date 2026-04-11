// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DomainProfile {
    pub primary_domain: String,
    pub kind: String,
    pub capabilities: Vec<String>,
}

impl DomainProfile {
    pub fn underwater() -> Self { Self { primary_domain: "underwater".into(), kind: "underwater".into(), capabilities: vec!["sonar".into()] } }
    pub fn subterranean() -> Self { Self { primary_domain: "subterranean".into(), kind: "subterranean".into(), capabilities: vec!["lidar".into()] } }
    pub fn deep_space() -> Self { Self { primary_domain: "deep_space".into(), kind: "deep_space".into(), capabilities: vec!["star_tracker".into()] } }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlatformCapabilityProfile {
    pub manipulation: f64, pub locomotion: f64, pub perception: f64, pub communication: f64,
}

impl PlatformCapabilityProfile {
    pub fn for_platform(platform: symthaea_core::embodiment::EmbodimentPlatform) -> Self {
        use symthaea_core::embodiment::EmbodimentPlatform;
        match platform {
            EmbodimentPlatform::Humanoid => Self { manipulation: 0.8, locomotion: 0.9, perception: 0.7, communication: 0.5 },
            EmbodimentPlatform::Quadrotor => Self { manipulation: 0.0, locomotion: 0.9, perception: 0.8, communication: 0.7 },
            EmbodimentPlatform::Vehicle => Self { manipulation: 0.0, locomotion: 1.0, perception: 0.6, communication: 0.8 },
            EmbodimentPlatform::Helicopter => Self { manipulation: 0.0, locomotion: 0.95, perception: 0.7, communication: 0.6 },
            EmbodimentPlatform::Auv => Self { manipulation: 0.3, locomotion: 0.7, perception: 0.5, communication: 0.2 },
            EmbodimentPlatform::Manipulator => Self { manipulation: 1.0, locomotion: 0.0, perception: 0.6, communication: 0.4 },
            EmbodimentPlatform::Exoskeleton => Self { manipulation: 0.5, locomotion: 0.8, perception: 0.3, communication: 0.3 },
            EmbodimentPlatform::Surgical => Self { manipulation: 1.0, locomotion: 0.0, perception: 0.9, communication: 0.5 },
            EmbodimentPlatform::Orbital => Self { manipulation: 0.9, locomotion: 0.0, perception: 0.4, communication: 0.3 },
            EmbodimentPlatform::Quadruped => Self { manipulation: 0.0, locomotion: 0.85, perception: 0.7, communication: 0.5 },
            _ => Self::default(),
        }
    }
    pub fn supports_domain(&self, _domain: &str) -> bool { true }
    pub fn preferred_domain_profile(&self) -> DomainProfile { DomainProfile::default() }
}
