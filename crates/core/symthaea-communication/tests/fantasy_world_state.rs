// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/fantasy_world_state.rs"]
mod fantasy_world_state;

use fantasy_world_state::*;

#[test]
fn fictional_history_can_change_without_rewriting_prior_facts() {
    let mut world = FantasyWorldStateV1::new(
        "world-a",
        "session-a",
        FantasySceneStateV1::new("scene-1", 1, None).unwrap(),
    )
    .unwrap();

    world
        .record_fact(
            FantasyFactV1::new(
                "fact-1",
                "setting.location",
                "fiction:place-one",
                FantasyFactSourceV1::ParticipantAuthored,
                1,
                None,
            )
            .unwrap(),
        )
        .unwrap();
    world
        .record_fact(
            FantasyFactV1::new(
                "fact-2",
                "setting.location",
                "fiction:place-two",
                FantasyFactSourceV1::SharedNarrative,
                2,
                Some("fact-1".into()),
            )
            .unwrap(),
        )
        .unwrap();

    let current = world.current_fact_for_key("setting.location").unwrap();
    assert_eq!(current.fact_id(), "fact-2");
    assert_eq!(current.value_ref(), "fiction:place-two");
    assert_eq!(world.namespace(), FantasyRealityNamespaceV1::FictionOnly);
}

#[test]
fn narrative_thread_and_callback_state_survive_scene_progression() {
    let mut world = FantasyWorldStateV1::new(
        "world-a",
        "session-a",
        FantasySceneStateV1::new("scene-1", 1, None).unwrap(),
    )
    .unwrap();
    world
        .open_thread(FantasyNarrativeThreadV1::new("thread-a", "fiction:thread-a", 1).unwrap())
        .unwrap();
    world.add_callback_ref("fiction:callback-a").unwrap();
    world
        .advance_scene(FantasySceneStateV1::new("scene-2", 2, None).unwrap())
        .unwrap();

    assert_eq!(world.thread("thread-a").unwrap().state(), FantasyThreadStateV1::Open);
    assert!(world.callback_refs().contains("fiction:callback-a"));
}

#[test]
fn participant_and_symthaea_authorship_remain_distinguishable() {
    let participant = FantasyFactV1::new(
        "fact-p",
        "character.preference",
        "fiction:value-p",
        FantasyFactSourceV1::ParticipantAuthored,
        1,
        None,
    )
    .unwrap();
    let symthaea = FantasyFactV1::new(
        "fact-s",
        "character.idea",
        "fiction:value-s",
        FantasyFactSourceV1::SymthaeaAuthored,
        1,
        None,
    )
    .unwrap();

    assert_eq!(participant.source(), FantasyFactSourceV1::ParticipantAuthored);
    assert_eq!(symthaea.source(), FantasyFactSourceV1::SymthaeaAuthored);
}
