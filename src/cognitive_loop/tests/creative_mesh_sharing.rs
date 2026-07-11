// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test for cross-agent culture sharing (ART_CULTURE_REVIEW_AND_PLAN_2026-07-06,
//! Phase 4): verifies `CreativeManager` actually ticks live inside the running cognitive loop
//! (it previously had zero call sites outside its own unit tests) and that a fresh published
//! artifact produces a real `ContentAnnounce`-typed `MeshOutbound` packet on the mesh outbound
//! channel — not just that the wiring compiles.

#![cfg(all(feature = "creative", feature = "social-fabric"))]

use super::super::{CognitiveLoopConfig, CognitiveLoopService};
use crate::swarm::mesh::PayloadType;

#[test]
fn creative_manager_ticks_live_and_publishes() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let inputs = [
        "consciousness emerges from integration",
        "the garden grows in silence",
        "hello world, what do you see",
        "a memory of light and water",
        "we are learning to trust each other",
    ];

    let mut saw_artwork = false;
    for i in 0..300 {
        let input = inputs[i % inputs.len()];
        let _ = service.cycle(input);

        let total = service
            .sensorimotor
            .motor_rendering
            .creative_manager
            .as_ref()
            .expect("creative_manager present under `creative` feature")
            .last_telemetry()
            .total_artworks;
        if total > 0 {
            saw_artwork = true;
            break;
        }
    }

    assert!(
        saw_artwork,
        "CreativeManager never produced an artwork across 300 cycles — \
         the manager is not actually ticking inside the live cognitive loop"
    );

    let rx = service
        .take_mesh_outbound_rx()
        .expect("mesh outbound receiver must be available before first take");

    let mut saw_content_announce = false;
    while let Ok(outbound) = rx.try_recv() {
        if outbound.packet.payload_type == PayloadType::ContentAnnounce {
            saw_content_announce = true;
            break;
        }
    }

    assert!(
        saw_content_announce,
        "no ContentAnnounce packet was sent on the mesh outbound channel despite \
         CreativeManager publishing at least one artifact — cross-agent culture \
         sharing send-side wiring is not firing"
    );
}
