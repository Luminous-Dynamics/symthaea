// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Skeletal animation and IK integration for robotics platforms.

use crate::plugin::SymtropyPhysics;
use bevy::prelude::*;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_robotics_bridge_core::platform::PlatformType;

/// Current animation state for a robotic agent.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Component)]
pub enum AnimationState {
    #[default]
    Idle,
    Walk,
    Run,
    Stumble,
    Active, // For manipulators
    Error,  // Red tier / low phi
}

/// Component identifying an entity as an animation target (e.g., a specific joint or effector).
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct AnimationTarget {
    pub platform: PlatformType,
    pub index: usize,
}

pub struct RoboticAnimationPlugin;

impl Plugin for RoboticAnimationPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AnimationState>()
            .register_type::<AnimationTarget>()
            .add_systems(
                Update,
                (
                    phi_gated_animation_system::<2>,
                    phi_gated_animation_system::<3>,
                    phi_gated_animation_system::<4>,
                ),
            );
    }
}

/// Update animation state based on current Φ and safety tier.
pub fn phi_gated_animation_system<const D: usize>(
    physics: Res<SymtropyPhysics<D>>,
    mut query: Query<(&crate::PhysicsBody, &mut AnimationState)>,
) {
    for (body, mut state) in &mut query {
        let phi = physics.field.phi(body.handle);
        let tier = physics.field.safety_tier(body.handle);

        let target_state = match tier {
            SafetyTier::Red => AnimationState::Error,
            SafetyTier::Orange => AnimationState::Stumble,
            SafetyTier::Yellow => {
                if phi < 0.15 {
                    AnimationState::Stumble
                } else {
                    AnimationState::Walk
                }
            }
            SafetyTier::Green => {
                if phi > 0.8 {
                    AnimationState::Run
                } else {
                    AnimationState::Walk
                }
            }
        };

        if *state != target_state {
            *state = target_state;
        }
    }
}
