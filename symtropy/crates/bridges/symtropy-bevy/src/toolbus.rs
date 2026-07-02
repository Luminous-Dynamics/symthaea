// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Toolhead management for the symthaea-toolbus.
//! Handles physical merging, cognitive prior injection, and energy drain.

use bevy::prelude::*;
use symthaea_bevy_brain::CognitiveBrain;
use symtropy_bevy_core::PhysicsBody;
use symtropy_math::Transform;
use symtropy_math::shape::Shape;

/// Component indicating a toolhead is ready to be locked.
#[derive(Component)]
pub struct ToolheadAttachment {
    pub tool_mass: f64,
    pub local_transform: Transform<3>,
    pub power_drain_watts: f64,
    pub tool_collider: Box<dyn Shape<3>>,
}

/// Event triggered when a toolhead lock is confirmed (DID Auth success).
#[derive(Debug, Clone, Message)]
pub struct ToolheadLockedEvent {
    pub host_entity: Entity,
    pub tool_entity: Entity,
}

/// System to handle the "Passport Route" (Option B) for toolhead engagement.
pub fn toolhead_lock_system(
    mut events: MessageReader<ToolheadLockedEvent>,
    mut host_query: Query<(&mut PhysicsBody, &CognitiveBrain)>,
    tool_query: Query<&ToolheadAttachment>,
    mut physics: ResMut<crate::plugin::SymtropyPhysics<3>>,
) {
    for event in events.read() {
        if let (Ok((host_body, brain)), Ok(tool)) = (
            host_query.get_mut(event.host_entity),
            tool_query.get(event.tool_entity),
        ) {
            // 1. Physics Transition: Merge toolhead into host body
            if let Some(rb) = physics.world.body_mut(host_body.handle) {
                rb.merge_toolhead(
                    tool.local_transform.clone(),
                    tool.tool_collider.clone_box(),
                    tool.tool_mass,
                );
                info!("Physics: Merged toolhead into body {:?}", host_body.handle);
            }

            // 2. Cognitive Transition: Passport Route (Direct Prior Injection)
            let target_mean = vec![0.8; 10];
            let target_precision = vec![2.0; 10];
            brain.inject_priors(target_mean, target_precision);
            info!(
                "Cognition: Injected toolhead priors into brain for entity {:?}",
                event.host_entity
            );

            // 3. Thermodynamic Transition: Add power drain
            if let Some(entity_state) = physics.field.entities.get_mut(&host_body.handle) {
                entity_state.power_drain_watts = tool.power_drain_watts;
                info!(
                    "Thermodynamics: Registered {}W drain for entity {:?}",
                    tool.power_drain_watts, event.host_entity
                );
            }
        }
    }
}

/// System that penalizes moral resonance if power usage is high relative to Phi.
pub fn wastefulness_monitor_system(
    physics: Res<crate::plugin::SymtropyPhysics<3>>,
    mut query: Query<(&PhysicsBody, &mut CognitiveBrain)>,
) {
    for (body, mut brain) in &mut query {
        if let Some(entity_state) = physics.field.entities.get(&body.handle) {
            let power = entity_state.power_drain_watts;
            let phi = brain.phi();

            if power > 0.0 {
                // Wastefulness = Power / Phi (High power with low Phi is "wasteful")
                let wastefulness = (power / (phi + 0.1)).min(10.0);

                // Penalize stewardship_care based on wastefulness
                let penalty = (wastefulness * 0.01) as f64;
                brain.profile.stewardship_care =
                    (brain.profile.stewardship_care - penalty).max(0.0);

                if penalty > 0.05 {
                    warn!(
                        "NPC-{:?} is being WASTEFUL: Stewardship Care dropping!",
                        body.handle
                    );
                }
            }
        }
    }
}
