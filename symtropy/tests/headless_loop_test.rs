// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use symtropy_launcher::plugin::SymtropyPlugin;
use symtropy_launcher::resources::GamePhase;

#[test]
fn test_headless_game_loop() {
    let mut app = App::new();

    // Set up standard Bevy plugins headlessly
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.add_plugins(bevy::input::InputPlugin);
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.init_asset::<Mesh>();
    app.init_asset::<StandardMaterial>();

    // Register our plugin (which handles systems, state, physics, settlement loop)
    app.add_plugins(SymtropyPlugin);

    // Selection experience setup: select "waterworks-3d"
    {
        let mut registry = app
            .world_mut()
            .resource_mut::<symtropy_launcher::experience::ExperienceRegistry>();
        if let Some(idx) = registry
            .experiences
            .iter()
            .position(|e| e.id == "waterworks-3d")
        {
            registry.selected = idx;
        }
    }

    // Transition from MainMenu to Loading manually
    app.world_mut()
        .resource_mut::<NextState<GamePhase>>()
        .set(GamePhase::Loading);

    // Step 1: Let Bevy run the state transition to Loading, which triggers procgen and setup_world systems
    app.update();

    // Step 2: Next frame update lets the state transition system run, moving the state from Loading to Playing/Playing3D (thanks to auto_start)
    app.update();

    // Step 3: Transition to Playing3D / Playing. Run OnEnter systems for the active gameplay phase.
    app.update();

    // Run for 100+ frames to simulate gameplay ticks
    for _ in 0..120 {
        app.update();
    }

    // --- Assertions ---

    // 1. Verify Player is spawned
    let mut player_query = app
        .world_mut()
        .query::<&symtropy_launcher::components::Player>();
    let player_count = player_query.iter(app.world()).count();
    assert_eq!(player_count, 1, "Player entity must be spawned");

    // 2. Verify all 7 named NPCs are spawned
    let mut npc_query = app
        .world_mut()
        .query::<&symtropy_launcher::components::CrewNpc>();
    let npcs: Vec<_> = npc_query.iter(app.world()).collect();
    assert_eq!(npcs.len(), 7, "All 7 crew NPCs must be spawned");

    // 3. Verify NPC names
    let expected_names = [
        "Engineer (Kael)",
        "Medic (Mira)",
        "Archivist (Soren)",
        "Convoy Lead (Jack)",
        "Friendly Robot (PR-4)",
        "Industrial Liaison (Nadia)",
        "Young Tech (Leo)",
    ];
    for name in &expected_names {
        assert!(
            npcs.iter().any(|npc| npc.name == *name),
            "Crew NPC {} is missing",
            name
        );
    }

    // 4. Verify Active Inference loops are ticking and updating NPC states / targets
    // Every NPC has a MoveTarget and PsychologicalNeeds.
    let mut psych_query = app
        .world_mut()
        .query::<&symtropy_launcher::systems::psychology::PsychologicalNeeds>();
    let psychs: Vec<_> = psych_query.iter(app.world()).collect();
    assert_eq!(
        psychs.len(),
        7,
        "All 7 NPCs must have psychological needs component"
    );

    let mut move_target_query = app
        .world_mut()
        .query::<&symtropy_launcher::components::MoveTarget>();
    let move_targets: Vec<_> = move_target_query.iter(app.world()).collect();
    assert_eq!(move_targets.len(), 7, "All 7 NPCs must have move targets");
    let has_non_empty_targets = move_targets.iter().any(|t| t.target.is_some());
    assert!(
        has_non_empty_targets,
        "At least some NPCs should have calculated move targets via FEP perception-action loops"
    );

    // 5. Verify Settlement Metrics change/exist
    let settlement_metrics = app
        .world()
        .get_resource::<symtropy_launcher::resources::SettlementMetrics>()
        .cloned()
        .expect("SettlementMetrics resource must be initialized");
    assert!(settlement_metrics.power >= 0.0);

    // 6. Test NullDrone behavior
    // Verify there are no drones initially
    let drone_count_init = app
        .world_mut()
        .query::<&symtropy_launcher::components::NullDrone>()
        .iter(app.world())
        .count();
    assert_eq!(drone_count_init, 0, "No Null Drones should exist initially");

    // Spawn a NullDrone with a Transform
    let drone_entity = app
        .world_mut()
        .spawn((
            symtropy_launcher::components::NullDrone::default(),
            Transform::default(),
        ))
        .id();

    // Query existing PowerJunction target machine and repair it first so the drone wants to target it
    let junction_entity = {
        let mut query = app
            .world_mut()
            .query::<(Entity, &mut symtropy_launcher::components::PowerJunction)>();
        let (j_ent, mut junction) = query
            .iter_mut(app.world_mut())
            .next()
            .expect("At least one PowerJunction should be spawned");
        junction.is_damaged = false;
        j_ent
    };

    // Run update tick to trigger NullDrone AI targeting
    app.update();

    // Assert the NullDrone has chosen a target machine
    let drone = app
        .world()
        .get::<symtropy_launcher::components::NullDrone>(drone_entity)
        .expect("NullDrone entity must exist");
    assert!(
        drone.target_machine.is_some(),
        "Null Drone AI must select a target machine"
    );

    // Simulate drone reaching the target machine (teleport it to the target's position and run sabotage tick)
    let target_translation = app
        .world()
        .get::<Transform>(junction_entity)
        .expect("PowerJunction must have a Transform")
        .translation;

    app.world_mut()
        .get_mut::<Transform>(drone_entity)
        .expect("NullDrone must have a Transform")
        .translation = target_translation;

    // Tick the app to execute sabotage logic in null_drone_ai_system
    app.update();

    // Verify the target machine was damaged / sabotaged
    let junction_after = app
        .world()
        .get::<symtropy_launcher::components::PowerJunction>(junction_entity)
        .expect("PowerJunction must still exist");
    assert!(
        junction_after.is_damaged,
        "Target PowerJunction must be damaged after drone sabotage"
    );

    println!(
        "Headless simulation completed successfully with drone sabotage validation. Settlement metrics: {:?}",
        settlement_metrics
    );
}
