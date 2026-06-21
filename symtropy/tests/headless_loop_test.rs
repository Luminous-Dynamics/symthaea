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
    app.add_plugins(bevy::gizmos::GizmoPlugin);
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

    // Measure duration using std::time::Instant
    let start_time = std::time::Instant::now();

    // Run for 100+ frames to simulate gameplay ticks
    for _ in 0..120 {
        app.update();
    }

    let elapsed = start_time.elapsed();
    println!("120-frame simulation took: {:?}", elapsed);
    assert!(
        elapsed < std::time::Duration::from_secs_f32(3.0),
        "Headless execution budget exceeded: 120 frames took {:?} (budget: 3.0s)",
        elapsed
    );

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

#[test]
fn test_dungeon_pcg_properties() {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for i in 0..50 {
        let seed = rng.r#gen::<u64>();
        // Randomize dimensions
        let width = rng.gen_range(20..50);
        let height = rng.gen_range(20..50);
        let dungeon = symtropy_launcher::systems::procgen::generate_dungeon(width, height, seed);

        // Verify dimensions
        assert_eq!(dungeon.width, width);
        assert_eq!(dungeon.height, height);

        // Find player start (3) and core start (2)
        let mut player_start = None;
        let mut core_start = None;

        for y in 0..height {
            for x in 0..width {
                if dungeon.tiles[y][x] == 3 {
                    player_start = Some((x, y));
                } else if dungeon.tiles[y][x] == 2 {
                    core_start = Some((x, y));
                }
            }
        }

        assert!(
            player_start.is_some(),
            "Dungeon layout #{} (seed: {}) failed to generate a player start",
            i,
            seed
        );
        assert!(
            core_start.is_some(),
            "Dungeon layout #{} (seed: {}) failed to generate a core start",
            i,
            seed
        );

        let start_pos = player_start.unwrap();
        let end_pos = core_start.unwrap();

        // Perform BFS to assert that a path exists between them
        let mut queue = std::collections::VecDeque::new();
        let mut visited = vec![vec![false; width]; height];

        queue.push_back(start_pos);
        visited[start_pos.1][start_pos.0] = true;

        let mut path_found = false;
        while let Some((cx, cy)) = queue.pop_front() {
            if (cx, cy) == end_pos {
                path_found = true;
                break;
            }

            for (dx, dy) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                    let ux = nx as usize;
                    let uy = ny as usize;
                    // Walkable tiles are > 0 (1=floor, 2=core_room, 3=player_start)
                    if !visited[uy][ux] && dungeon.tiles[uy][ux] > 0 {
                        visited[uy][ux] = true;
                        queue.push_back((ux, uy));
                    }
                }
            }
        }

        assert!(
            path_found,
            "Dungeon layout #{} (seed: {}) failed path verification: no walkable path exists between player start {:?} and core start {:?}",
            i, seed, start_pos, end_pos
        );
    }
}

#[test]
fn test_narrative_system_causality() {
    let mut app = App::new();

    // Set up standard Bevy plugins headlessly
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.add_plugins(bevy::input::InputPlugin);
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.add_plugins(bevy::gizmos::GizmoPlugin);
    app.init_asset::<Mesh>();
    app.init_asset::<StandardMaterial>();

    // Register our plugin (which handles systems, state, physics, settlement loop)
    app.add_plugins(SymtropyPlugin);

    // Mock virtual time advancement by 0.5s per frame deterministically
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f32(0.5),
    ));

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

    // Run setup to playing transition
    app.update();
    app.update();
    app.update();

    // Now, let's create a test event collector resource and add the collector system
    #[derive(Resource, Default, Clone)]
    struct TestEventCollector {
        actions: Vec<symtropy_launcher::components::NpcActionEvent>,
        feedback: Vec<symtropy_launcher::components::WorldFeedbackEvent>,
    }

    app.insert_resource(TestEventCollector::default());

    fn collect_test_events_system(
        mut reader_actions: MessageReader<symtropy_launcher::components::NpcActionEvent>,
        mut reader_feedback: MessageReader<symtropy_launcher::components::WorldFeedbackEvent>,
        mut collector: ResMut<TestEventCollector>,
    ) {
        for event in reader_actions.read() {
            collector.actions.push(event.clone());
        }
        for event in reader_feedback.read() {
            collector.feedback.push(event.clone());
        }
    }

    app.add_systems(Update, collect_test_events_system);

    // Helper to teleport both the Bevy Transform and Rapier physics body of an NPC
    let teleport_npc = |app: &mut App, entity: Entity, pos: Vec3| {
        if let Some(mut tf) = app.world_mut().get_mut::<Transform>(entity) {
            tf.translation = pos;
        }
        let handle = app
            .world()
            .get::<symtropy_render_bridge::PhysicsBody>(entity)
            .map(|pb| pb.handle);
        if let Some(handle) = handle {
            let mut physics = app
                .world_mut()
                .resource_mut::<symtropy_launcher::resources::PhysicsWorldRes>();
            if let Some(body) = physics.world.body_mut(handle) {
                body.transform.translation =
                    symtropy_math::Point::new([pos.x as f64, pos.y as f64]);
            }
        }
    };

    // Let's retrieve entities we want to manipulate
    let mut kael_entity = None;
    let mut pr4_entity = None;
    let mut soren_entity = None;
    let mut mira_entity = None;
    let mut leo_entity = None;
    let mut jack_entity = None;

    {
        let mut query = app
            .world_mut()
            .query::<(Entity, &symtropy_launcher::components::CrewNpc)>();
        for (entity, npc) in query.iter(app.world()) {
            if npc.name.contains("Kael") {
                kael_entity = Some(entity);
            } else if npc.name.contains("PR-4") {
                pr4_entity = Some(entity);
            } else if npc.name.contains("Soren") {
                soren_entity = Some(entity);
            } else if npc.name.contains("Mira") {
                mira_entity = Some(entity);
            } else if npc.name.contains("Leo") {
                leo_entity = Some(entity);
            } else if npc.name.contains("Jack") {
                jack_entity = Some(entity);
            }
        }
    }

    let kael = kael_entity.unwrap();
    let pr4 = pr4_entity.unwrap();
    let soren = soren_entity.unwrap();
    let mira = mira_entity.unwrap();
    let leo = leo_entity.unwrap();
    let jack = jack_entity.unwrap();

    // 1. Verify "when PR-4 reaches sabotaged pump -> pump damage decreases" and partial success
    // Let's spawn a sabotaged WaterPump
    let pump_pos = Vec3::new(100.0, 100.0, 0.0);
    let pump_entity = app
        .world_mut()
        .spawn((
            symtropy_launcher::components::WaterPump {
                efficiency: 0.0,
                is_running: false,
                is_sabotaged: true,
            },
            Transform::from_translation(pump_pos),
        ))
        .id();

    // Teleport PR-4 next to the pump, soren far away
    teleport_npc(&mut app, pr4, pump_pos);
    teleport_npc(&mut app, soren, Vec3::new(500.0, 500.0, 0.0));

    // Update app for a few frames (with dt simulation)
    for _ in 0..10 {
        app.update();
    }

    // Verify pump efficiency improved, is_running is true, is_sabotaged is false, but capped at 0.7
    let pump = app
        .world()
        .get::<symtropy_launcher::components::WaterPump>(pump_entity)
        .unwrap();
    assert!(pump.efficiency > 0.0);
    assert!(pump.efficiency <= 0.7);
    assert!(pump.is_running);
    assert!(!pump.is_sabotaged);

    // Run one more frame to ensure messages written in the 10th frame are collected in Update
    app.update();

    // Verify online but contaminated event was fired
    let collector = app.world().resource::<TestEventCollector>();
    assert!(
        collector
            .feedback
            .iter()
            .any(|f| f.message.contains("CONTAMINATED"))
    );

    // Teleport Soren near the pump too, sabotage it again, and run update
    app.world_mut()
        .get_mut::<symtropy_launcher::components::WaterPump>(pump_entity)
        .unwrap()
        .is_sabotaged = true;
    teleport_npc(&mut app, soren, pump_pos);

    // Clear collector for next step
    app.world_mut()
        .resource_mut::<TestEventCollector>()
        .feedback
        .clear();

    for _ in 0..10 {
        app.update();
    }

    // Verify full purification to 1.0 (cooperative success with Soren)
    let pump = app
        .world()
        .get::<symtropy_launcher::components::WaterPump>(pump_entity)
        .unwrap();
    assert_eq!(pump.efficiency, 1.0);
    assert!(!pump.is_sabotaged);

    let collector = app.world().resource::<TestEventCollector>();
    assert!(
        collector
            .feedback
            .iter()
            .any(|f| f.message.contains("100%"))
    );

    // 2. Verify "when pump damage decreases -> settlement water metric improves"
    let settlement_metrics = app
        .world()
        .resource::<symtropy_launcher::resources::SettlementMetrics>();
    assert!(settlement_metrics.water > 0.0);

    // 3. Verify "when Leo is near Mira and Kael is far away -> Leo relapses"
    // Set Leo's stress high (> 0.4)
    app.world_mut()
        .get_mut::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load = 0.5;

    // Teleport Mira next to Leo
    let leo_pos = Vec3::new(200.0, 200.0, 0.0);
    teleport_npc(&mut app, leo, leo_pos);
    teleport_npc(&mut app, mira, leo_pos);

    // Teleport Kael far away (>120 units)
    teleport_npc(&mut app, kael, Vec3::new(400.0, 400.0, 0.0));

    // Run updates
    app.world_mut()
        .resource_mut::<TestEventCollector>()
        .feedback
        .clear();
    for _ in 0..100 {
        app.update();
    }

    // Verify Leo relapses (stress increases instead of decays)
    let leo_needs = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap();
    assert!(leo_needs.allostatic_load > 0.5);

    // 4. Verify "when Kael is near Leo -> Leo stress decays under Mira's care"
    teleport_npc(&mut app, kael, leo_pos);
    let stress_before = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load;

    for _ in 0..50 {
        app.update();
    }

    let stress_after = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load;
    assert!(
        stress_after < stress_before,
        "Stress should decay when Kael is near"
    );

    // 5. Verify Jack drone destruction event emitting
    let drone_pos = Vec3::new(300.0, 300.0, 0.0);
    let drone_entity = app
        .world_mut()
        .spawn((
            symtropy_launcher::components::NullDrone::default(),
            Transform::from_translation(drone_pos),
        ))
        .id();

    // Teleport Jack next to the drone
    teleport_npc(&mut app, jack, drone_pos);

    app.world_mut()
        .resource_mut::<TestEventCollector>()
        .actions
        .clear();

    // Update to trigger combat and verify drone health decreases or despawns, and action event fires
    for _ in 0..10 {
        app.update();
    }

    let collector = app.world().resource::<TestEventCollector>();
    assert!(
        collector
            .actions
            .iter()
            .any(|a| a.action_kind == symtropy_launcher::components::NpcActionKind::CombatDrone)
    );
}

#[test]
fn test_old_waterworks_tutorial_causal_chain() {
    let mut app = App::new();

    // Set up standard Bevy plugins headlessly
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.add_plugins(bevy::input::InputPlugin);
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.add_plugins(bevy::gizmos::GizmoPlugin);
    app.init_asset::<Mesh>();
    app.init_asset::<StandardMaterial>();

    // Register our plugin (which handles systems, state, physics, settlement loop)
    app.add_plugins(SymtropyPlugin);

    // Mock virtual time advancement by 0.5s per frame deterministically
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f32(0.5),
    ));

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
        .resource_mut::<NextState<symtropy_launcher::resources::GamePhase>>()
        .set(symtropy_launcher::resources::GamePhase::Loading);

    // Run setup to playing transition
    app.update();
    app.update();
    app.update();

    // Now, let's create a test event collector resource and add the collector system
    #[derive(Resource, Default, Clone)]
    struct TestEventCollector {
        feedback: Vec<symtropy_launcher::components::WorldFeedbackEvent>,
    }

    app.insert_resource(TestEventCollector::default());

    fn collect_test_events_system(
        mut reader_feedback: MessageReader<symtropy_launcher::components::WorldFeedbackEvent>,
        mut collector: ResMut<TestEventCollector>,
    ) {
        for event in reader_feedback.read() {
            collector.feedback.push(event.clone());
        }
    }

    app.add_systems(Update, collect_test_events_system);

    // Helper to teleport both the Bevy Transform and Rapier physics body of an NPC
    let teleport_npc = |app: &mut App, entity: Entity, pos: Vec3| {
        if let Some(mut tf) = app.world_mut().get_mut::<Transform>(entity) {
            tf.translation = pos;
        }
        let handle = app
            .world()
            .get::<symtropy_render_bridge::PhysicsBody>(entity)
            .map(|pb| pb.handle);
        if let Some(handle) = handle {
            let mut physics = app
                .world_mut()
                .resource_mut::<symtropy_launcher::resources::PhysicsWorldRes>();
            if let Some(body) = physics.world.body_mut(handle) {
                body.transform.translation =
                    symtropy_math::Point::new([pos.x as f64, pos.y as f64]);
            }
        }
    };

    // Retrieve entities we want to manipulate
    let mut pr4_entity = None;
    let mut soren_entity = None;
    let mut kael_entity = None;
    let mut mira_entity = None;
    let mut leo_entity = None;

    {
        let mut query = app
            .world_mut()
            .query::<(Entity, &symtropy_launcher::components::CrewNpc)>();
        for (entity, npc) in query.iter(app.world()) {
            if npc.name.contains("PR-4") {
                pr4_entity = Some(entity);
            } else if npc.name.contains("Soren") {
                soren_entity = Some(entity);
            } else if npc.name.contains("Kael") {
                kael_entity = Some(entity);
            } else if npc.name.contains("Mira") {
                mira_entity = Some(entity);
            } else if npc.name.contains("Leo") {
                leo_entity = Some(entity);
            }
        }
    }

    let pr4 = pr4_entity.unwrap();
    let soren = soren_entity.unwrap();
    let kael = kael_entity.unwrap();
    let mira = mira_entity.unwrap();
    let leo = leo_entity.unwrap();

    // 1. Initial pump sabotaged.
    let pump_pos = Vec3::new(100.0, 100.0, 0.0);
    let pump_entity = app
        .world_mut()
        .spawn((
            symtropy_launcher::components::WaterPump {
                efficiency: 0.0,
                is_running: false,
                is_sabotaged: true,
            },
            Transform::from_translation(pump_pos),
        ))
        .id();

    // Initialize the tutorial scenario resource at step PumpSabotaged
    app.insert_resource(symtropy_launcher::resources::TutorialScenarioRes {
        step: symtropy_launcher::resources::TutorialStep::PumpSabotaged,
        pump_entity: Some(pump_entity),
    });

    // Teleport PR-4 next to the pump, Soren far away
    teleport_npc(&mut app, pr4, pump_pos);
    teleport_npc(&mut app, soren, Vec3::new(500.0, 500.0, 0.0));

    // Teleport Kael far away from Leo
    teleport_npc(&mut app, kael, Vec3::new(600.0, 600.0, 0.0));
    teleport_npc(&mut app, leo, Vec3::new(200.0, 200.0, 0.0));
    teleport_npc(&mut app, mira, Vec3::new(200.0, 200.0, 0.0));

    // Update app for a few frames. PR-4 should start repair, transitioning step to PR4Repairing
    app.update();
    app.update();

    let tutorial = app
        .world()
        .resource::<symtropy_launcher::resources::TutorialScenarioRes>();
    assert_eq!(
        tutorial.step,
        symtropy_launcher::resources::TutorialStep::PR4Repairing
    );

    // 2. PR-4 alone reaches 70% efficiency online (contaminated)
    // Run update frames until efficiency reaches 0.7.
    // At that point, the tutorial system will transition to CoopRepairing.
    for _ in 0..10 {
        app.update();
    }

    let pump = app
        .world()
        .get::<symtropy_launcher::components::WaterPump>(pump_entity)
        .unwrap();
    assert!(pump.efficiency >= 0.7);
    assert!(!pump.is_sabotaged); // partial online

    let tutorial = app
        .world()
        .resource::<symtropy_launcher::resources::TutorialScenarioRes>();
    assert_eq!(
        tutorial.step,
        symtropy_launcher::resources::TutorialStep::CoopRepairing
    );

    // 3. Verify contamination warning message generated
    app.update(); // propagate messages to reader
    let collector = app.world().resource::<TestEventCollector>();
    assert!(
        collector
            .feedback
            .iter()
            .any(|f| f.message.contains("CONTAMINATION"))
    );

    // 4. Soren joins PR-4 to cooperative repair
    // Teleport Soren next to the pump
    teleport_npc(&mut app, soren, pump_pos);

    // Run updates to let Soren and PR-4 cooperatively repair pump to 100% (1.0)
    let mut reached_rising = false;
    for _ in 0..25 {
        app.update();
        let tutorial = app
            .world()
            .resource::<symtropy_launcher::resources::TutorialScenarioRes>();
        if tutorial.step == symtropy_launcher::resources::TutorialStep::LeoStressRising {
            reached_rising = true;
            break;
        }
    }
    assert!(reached_rising, "Must transition to LeoStressRising step");

    let pump = app
        .world()
        .get::<symtropy_launcher::components::WaterPump>(pump_entity)
        .unwrap();
    assert_eq!(pump.efficiency, 1.0);
    assert!(!pump.is_sabotaged);

    // 5. Verify Coop success feedback fired
    app.update(); // propagate messages to reader
    let collector = app.world().resource::<TestEventCollector>();
    assert!(
        collector
            .feedback
            .iter()
            .any(|f| f.message.contains("COOPERATIVE REPAIR"))
    );

    // 6. Leo stress rises when Kael is distant
    // Now Leo's stress has just been set to 0.55.
    let stress_before = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load;
    assert!(
        stress_before >= 0.55 && stress_before < 0.8,
        "stress_before was {}",
        stress_before
    );

    for _ in 0..5 {
        app.update();
    }
    let stress_after = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load;
    assert!(
        stress_after > stress_before,
        "Leo's stress must rise when Kael is distant"
    );

    // 7. Mira stabilizes Leo when Kael returns
    // Teleport Kael next to Leo
    teleport_npc(&mut app, kael, Vec3::new(200.0, 200.0, 0.0));

    // Run updates. Leo's stress should start decaying
    let stress_high = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load;
    for _ in 0..15 {
        app.update();
    }
    let stress_low = app
        .world()
        .get::<symtropy_launcher::systems::psychology::PsychologicalNeeds>(leo)
        .unwrap()
        .allostatic_load;
    assert!(
        stress_low < stress_high,
        "Leo's stress must decrease when Kael is near under Mira's care"
    );

    // 8. WorldFeedbackEvent confirms settlement recovery and step is Completed
    for _ in 0..30 {
        app.update();
        let tutorial = app
            .world()
            .resource::<symtropy_launcher::resources::TutorialScenarioRes>();
        if tutorial.step == symtropy_launcher::resources::TutorialStep::Completed {
            break;
        }
    }

    let tutorial = app
        .world()
        .resource::<symtropy_launcher::resources::TutorialScenarioRes>();
    assert_eq!(
        tutorial.step,
        symtropy_launcher::resources::TutorialStep::Completed
    );

    app.update(); // propagate completion feedback event
    let collector = app.world().resource::<TestEventCollector>();
    assert!(
        collector
            .feedback
            .iter()
            .any(|f| f.message.contains("SETTLEMENT STABILIZED"))
    );
}
