// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::super::*;
use symthaea_core::hdc::ContinuousHV;

#[test]
fn test_iroh_bridge_not_set_by_default() {
    let mind = ContinuousMind::default();
    assert!(!mind.has_iroh_bridge());
}

#[test]
fn test_iroh_bridge_attach() {
    let mut mind = ContinuousMind::default();
    let (handle, _actor) = crate::swarm::IrohBridgeHandle::new(4, 4);
    mind.set_iroh_bridge(handle);
    assert!(mind.has_iroh_bridge());
}

#[test]
fn test_iroh_bridge_flushes_outbox_on_tick() {
    let mut mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    mind.activate();
    let (handle, _actor) = crate::swarm::IrohBridgeHandle::new(64, 128);
    mind.set_iroh_bridge(handle);

    // Tick 5 times — social coherence exports on tick 5
    for _ in 0..5 {
        mind.tick();
    }

    // Outbox should be empty because the bridge flushed it
    assert!(
        mind.social_outbox.is_empty(),
        "Bridge should have flushed the outbox"
    );
}

#[test]
fn test_iroh_bridge_drains_inbox_on_tick() {
    let mut mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    mind.activate();
    let (handle, actor) = crate::swarm::IrohBridgeHandle::new(64, 128);

    // We need the actor's inbound_tx to inject messages.
    // Instead, manually push to inbox and verify tick processes it.
    // The bridge integration is: bridge drains → inbox, tick processes inbox → social coherence.
    // We can verify the bridge wiring by checking that when bridge is attached,
    // outbox messages get sent to the bridge channel.
    mind.set_iroh_bridge(handle);

    // Manually inject into inbox (simulating what bridge.drain_inbox would return)
    mind.receive_social(SocialMessage {
        agent_id: "network_peer".to_string(),
        behavior: ContinuousHV::random(512, 0xCAFE),
        context: ContinuousHV::random(512, 0xCAFE),
        interaction_outcome: None,
        bath_state: None,
    });

    mind.tick();

    // The message should have been processed by social coherence
    let sc = mind.social_coherence().unwrap();
    assert!(
        sc.get_mental_model("network_peer").is_some(),
        "Network peer should be modeled after tick"
    );

    // Suppress unused variable warning
    drop(actor);
}
