// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::super::*;
use symthaea_core::hdc::ContinuousHV;

#[test]
fn test_social_coherence_disabled_by_default() {
    let mind = ContinuousMind::default();
    assert!(!mind.is_social());
    assert!(mind.social_coherence().is_none());
}

#[test]
fn test_social_coherence_enabled_via_config() {
    let mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    assert!(mind.is_social());
    assert!(mind.social_coherence().is_some());
}

#[test]
fn test_social_coherence_enable_after_construction() {
    let mut mind = ContinuousMind::default();
    assert!(!mind.is_social());
    mind.enable_social_coherence();
    assert!(mind.is_social());
}

#[test]
fn test_social_inbox_processed_on_tick() {
    let mut mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    mind.activate();

    // Send a social message
    mind.receive_social(SocialMessage {
        agent_id: "peer_1".to_string(),
        behavior: ContinuousHV::random(512, 0xBEEF_0001),
        context: ContinuousHV::random(512, 0xBEEF_0002),
        interaction_outcome: None,
        bath_state: None,
        #[cfg(feature = "swarm")]
        swarm_state: None,
    });

    assert_eq!(mind.social_inbox.len(), 1);
    mind.tick();
    // Inbox should be drained after tick
    assert_eq!(mind.social_inbox.len(), 0);
    // Agent should be modeled now
    let sc = mind.social_coherence().unwrap();
    assert!(sc.get_mental_model("peer_1").is_some());
}

#[test]
fn test_social_interaction_builds_relationship() {
    let mut mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    mind.activate();

    // Send cooperative interaction
    mind.receive_social(SocialMessage {
        agent_id: "ally_1".to_string(),
        behavior: ContinuousHV::random(512, 0xA11E_0001),
        context: ContinuousHV::random(512, 0xA11E_0002),
        interaction_outcome: Some(0.9),
        bath_state: None,
        #[cfg(feature = "swarm")]
        swarm_state: None,
    });
    mind.tick();

    let sc = mind.social_coherence().unwrap();
    let rel = sc.get_relationship("ally_1");
    assert!(rel.is_some(), "Relationship should be created");
    assert!(
        rel.unwrap().trust > 0.5,
        "Trust should increase from cooperation"
    );
}

#[test]
fn test_social_outbox_populated_on_tick() {
    let mut mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    mind.activate();

    // Tick 5 times to trigger outbox export (every 5 ticks)
    for _ in 0..5 {
        mind.tick();
    }

    let outbox = mind.drain_social_outbox();
    assert!(
        !outbox.is_empty(),
        "Outbox should have messages after 5 ticks"
    );
    assert_eq!(outbox[0].agent_id, "self");
}

#[test]
fn test_social_no_processing_when_disabled() {
    let mut mind = ContinuousMind::default();
    mind.activate();

    // Inbox messages should remain when social is disabled
    // (actually they get drained regardless but social coherence isn't updated)
    mind.receive_social(SocialMessage {
        agent_id: "ghost".to_string(),
        behavior: ContinuousHV::random(512, 0xDEAD),
        context: ContinuousHV::random(512, 0xDEAD),
        interaction_outcome: None,
        bath_state: None,
        #[cfg(feature = "swarm")]
        swarm_state: None,
    });
    mind.tick();

    // Social coherence is None, so no models are built
    assert!(mind.social_coherence().is_none());
    // Outbox should be empty since social is disabled
    let outbox = mind.drain_social_outbox();
    assert!(outbox.is_empty());
}

#[test]
fn test_social_outbox_capped() {
    let mut mind = ContinuousMind::new(MindConfig {
        enable_social_coherence: true,
        ..Default::default()
    });
    mind.activate();

    // Tick 1000 times — exports social every 5 ticks = 200 pushes
    for _ in 0..1000 {
        mind.tick();
    }

    assert!(
        mind.social_outbox.len() <= super::super::MAX_OUTBOX_SIZE,
        "social_outbox should be capped at {}: got {}",
        super::super::MAX_OUTBOX_SIZE,
        mind.social_outbox.len()
    );
}
