// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use rand::Rng;
use symthaea_bevy_brain::{CognitiveBrain, SymthaeaBrainPlugin};
use symtropy_bevy::PhysicsBody;
use symtropy_bevy::plugin::{SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_math::Point;
use symtropy_mycelix_bridge::{BevyMycelixPlugin, MycelixClient, MycelixRequest, MycelixResponse};

/// The current state of the village scenario.
#[derive(Resource, Default)]
struct VillageScenarioState {
    pub active_proposal_id: Option<String>,
    pub votes_cast: u32,
    pub scenario_timer: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "M4 Mycelix Village - Canonical 8D Profile".into(),
                ..default()
            }),
            ..default()
        }))
        // 1. Core Physics & Cognition
        .add_plugins(SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
        .add_plugins(SymthaeaBrainPlugin::default())
        // 2. The Subprocess IPC Bridge to Holochain
        .add_plugins(BevyMycelixPlugin::default())
        .init_resource::<VillageScenarioState>()
        .add_systems(Startup, setup_village)
        .add_systems(
            Update,
            (
                npc_mycelix_metabolism_system,
                scenario_driver_system,
                response_handler_system,
                visual_feedback_system,
            ),
        )
        .run();
}

fn setup_village(
    mut commands: Commands,
    mut physics: ResMut<SymtropyPhysics<3>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera and Light
    commands
        .spawn(Camera3d::default())
        .insert(Transform::from_xyz(0.0, 70.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y));
    commands.spawn(DirectionalLight {
        illuminance: 12000.0,
        shadows_enabled: true,
        ..default()
    });

    // Spawn 50 NPCs in a large radial formation
    let mut rng = rand::thread_rng();
    let mesh = meshes.add(Sphere::new(0.5).mesh().uv(16, 16));
    let mat = materials.add(Color::srgb(0.2, 0.7, 0.4));

    for i in 0..50 {
        let angle = (i as f32 / 50.0) * std::f32::consts::TAU;
        let radius = 35.0 + rng.gen_range(-10.0..10.0);
        let pos = Point::new([
            (angle.cos() * radius) as f64,
            0.5,
            (angle.sin() * radius) as f64,
        ]);

        let handle = physics.world.add_sphere(pos, 0.5, 1.0);
        physics.field.register(handle, 100.0, 10.0);

        commands.spawn((
            PhysicsBody::new(handle, 0.5),
            Mesh3d(mesh.clone()),
            MeshMaterial3d(mat.clone()),
            CognitiveBrain::new(64, &format!("village_npc_{}", i)),
            Name::new(format!("NPC-{}", i)),
        ));
    }

    // Ground plane
    let floor_mesh = meshes.add(Plane3d::default().mesh().size(250.0, 250.0));
    let floor_material = materials.add(Color::srgb(0.05, 0.08, 0.05));
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_material),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn npc_mycelix_metabolism_system(
    physics: Res<SymtropyPhysics<3>>,
    query: Query<(Entity, &PhysicsBody, &CognitiveBrain)>,
    mycelix_client: Res<MycelixClient>,
) {
    let mut rng = rand::thread_rng();
    for (entity, body, brain) in &query {
        let Some(entity_state) = physics.field.entities.get(&body.handle) else {
            continue;
        };
        let energy_fraction = entity_state.energy.fraction_remaining();

        // Use Domain Competence for TEND logic
        if energy_fraction < 0.3 && brain.profile.domain_competence > 0.4 && rng.gen_bool(0.005) {
            mycelix_client
                .send(MycelixRequest::QueryTendBalance {
                    requester: entity,
                    member_did: format!("did:key:agent_{}", body.handle.0),
                })
                .ok();
        }
    }
}

fn scenario_driver_system(
    time: Res<Time>,
    mut state: ResMut<VillageScenarioState>,
    query: Query<(Entity, &PhysicsBody, &CognitiveBrain)>,
    mycelix_client: Res<MycelixClient>,
) {
    state.scenario_timer += time.delta_secs();

    if state.active_proposal_id.is_none() && state.scenario_timer > 20.0 {
        state.scenario_timer = 0.0;

        // Leader: high Epistemic Integrity + high Civic Participation
        if let Some((entity, body, _brain)) = query.iter().max_by(|a, b| {
            (a.2.profile.epistemic_integrity + a.2.profile.civic_participation)
                .partial_cmp(&(b.2.profile.epistemic_integrity + b.2.profile.civic_participation))
                .unwrap()
        }) {
            let proposal_id = format!("SOVEREIGN-{}", rand::random::<u16>());
            mycelix_client
                .send(MycelixRequest::SubmitProposal {
                    requester: entity,
                    proposal_id: proposal_id.clone(),
                    title: "Civic Alignment Proposal".into(),
                    description: "Balancing thermodynamic yield against semantic resonance.".into(),
                    author_did: format!("did:key:agent_{}", body.handle.0),
                })
                .ok();

            state.active_proposal_id = Some(proposal_id);
            state.votes_cast = 0;
            info!("NPC-{} submitted Sovereign proposal!", body.handle.0);
        }
    }
}

fn response_handler_system(
    mut state: ResMut<VillageScenarioState>,
    mut reader: MessageReader<MycelixResponse>,
) {
    for response in reader.read() {
        match response {
            MycelixResponse::VoteCast { .. } => {
                state.votes_cast += 1;
                if state.votes_cast >= 15 {
                    state.active_proposal_id = None;
                }
            }
            _ => {}
        }
    }
}

/// Visual Feedback: NPC "Sovereign Radar" Gizmo based on the 8 canonical dimensions.
fn visual_feedback_system(mut gizmos: Gizmos, query: Query<(&Transform, &CognitiveBrain)>) {
    for (transform, brain) in &query {
        let pos = transform.translation + Vec3::Y * 0.5; // Slightly above ground
        let p = &brain.profile;

        // 8 Dimensions in the canonical order
        let factors = [
            p.epistemic_integrity,
            p.thermodynamic_yield,
            p.network_resilience,
            p.economic_velocity,
            p.civic_participation,
            p.stewardship_care,
            p.semantic_resonance,
            p.domain_competence,
        ];

        let color = Color::hsla(
            200.0,
            1.0,
            (p.epistemic_integrity as f32).clamp(0.2, 0.8),
            1.0,
        );

        for i in 0..8 {
            let angle = (i as f32 / 8.0) * std::f32::consts::TAU;
            let val = (factors[i] as f32 * 4.0).max(0.1);
            let end = pos + Vec3::new(angle.cos() * val, 0.0, angle.sin() * val);
            gizmos.line(pos, end, color);

            // Connect points to form the sovereign radar shape
            let next_angle = ((i + 1) as f32 / 8.0) * std::f32::consts::TAU;
            let next_val = (factors[(i + 1) % 8] as f32 * 4.0).max(0.1);
            let next_end = pos
                + Vec3::new(
                    next_angle.cos() * next_val,
                    0.0,
                    next_angle.sin() * next_val,
                );
            gizmos.line(end, next_end, color);
        }
    }
}
