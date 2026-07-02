// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D rendering setup and gameplay loop for the Embodied 3D Layer.

use crate::components::*;
use crate::resources::*;
use crate::systems::consciousness::NpcConsciousness;
use crate::systems::psychology::PsychologicalNeeds;
use crate::systems::rendering::LeviathanSprite;
use crate::systems::rendering::TILE_SIZE;
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::prelude::*;
use bevy::render::camera::CameraRenderGraph;
use bevy::render::view::NoIndirectDrawing;

/// Player marker for 3D physics and movement.
#[derive(Component)]
pub struct Player3D;

/// Main camera marker for 3D following.
#[derive(Component)]
pub struct Camera3D;

/// Floating prompt for the nearest actionable 3D interaction target.
#[derive(Component)]
pub struct InteractionFocusPrompt;

#[derive(Component)]
pub struct JunctionStatusBar;

#[derive(Component)]
pub struct JunctionEvidenceTag;

#[derive(Component)]
pub struct PumpSealBand;

#[derive(Component)]
pub struct PumpFlowLight;

#[derive(Component)]
pub struct FabricatorStatusPanel;

#[derive(Component)]
pub struct FabricatorEvidenceTag;

const PLAYER_COLLISION_RADIUS: f32 = 3.25;
const PLAYER_PHYSICS_RADIUS: f64 = 3.25;
const PLAYER_MOVE_SPEED: f32 = 85.0;
const PLAYER_VISUAL_HEIGHT: f32 = 7.5;
const NPC_COLLISION_RADIUS: f32 = 3.75;

const WET_CONCRETE: Color = Color::srgb(0.12, 0.14, 0.15);
const OLD_CONCRETE: Color = Color::srgb(0.28, 0.29, 0.27);
const RUSTED_METAL: Color = Color::srgb(0.46, 0.19, 0.08);
const PAINTED_STEEL: Color = Color::srgb(0.19, 0.31, 0.34);
const WARNING_AMBER: Color = Color::srgb(0.90, 0.62, 0.08);
const FIELD_DECK_CYAN: Color = Color::srgb(0.10, 0.70, 0.85);
const CORE_GOLD: Color = Color::srgb(0.95, 0.78, 0.15);
const LEVIATHAN_RED: Color = Color::srgb(0.90, 0.12, 0.08);
const DIRTY_WATER: Color = Color::srgb(0.05, 0.12, 0.14);
const ALGAE_STAIN: Color = Color::srgb(0.10, 0.22, 0.13);
const REPAIR_TAG: Color = Color::srgb(0.70, 0.90, 0.62);
const WORKWEAR_ORANGE: Color = Color::srgb(0.70, 0.34, 0.10);
const MEDIC_GREEN: Color = Color::srgb(0.18, 0.46, 0.34);
const ARCHIVE_BLUE: Color = Color::srgb(0.20, 0.25, 0.48);
const CONVOY_RED: Color = Color::srgb(0.48, 0.22, 0.20);
const ROBOT_TEAL: Color = Color::srgb(0.36, 0.54, 0.56);
const LIAISON_OCHRE: Color = Color::srgb(0.55, 0.50, 0.18);
const TECH_GREEN: Color = Color::srgb(0.38, 0.55, 0.20);
const RUBBER_BLACK: Color = Color::srgb(0.04, 0.045, 0.045);

fn waterworks_material(base_color: Color) -> StandardMaterial {
    StandardMaterial {
        base_color,
        unlit: true,
        perceptual_roughness: 0.95,
        reflectance: 0.0,
        ..default()
    }
}

/// Real-PBR variant of [`waterworks_material`] for surfaces upgraded from
/// flat greybox color to an actual CC0 texture (see assets/textures/waterworks/
/// and its ATTRIBUTION.md — ambientCG Concrete034/Concrete048/PaintedMetal004).
/// Deliberately lit (unlike the greybox default) since a flat-shaded texture
/// looks wrong; the scene's existing DirectionalLight provides real shading.
///
/// Deliberately does NOT wire ambientCG's standalone grayscale Roughness map
/// into `metallic_roughness_texture` — Bevy/glTF expect roughness in the
/// green channel and metallic in blue of one combined texture, so feeding in
/// a plain grayscale map (R=G=B) would make bright/rough areas incorrectly
/// read as partially metallic. Using a flat scalar `perceptual_roughness`
/// instead until there's a real reason to pack a proper ORM texture.
fn waterworks_textured_material(
    base_color_texture: Handle<Image>,
    normal_map_texture: Handle<Image>,
    tint: Color,
    perceptual_roughness: f32,
) -> StandardMaterial {
    StandardMaterial {
        base_color: tint,
        base_color_texture: Some(base_color_texture),
        normal_map_texture: Some(normal_map_texture),
        metallic: 0.0,
        perceptual_roughness,
        reflectance: 0.04,
        unlit: false,
        ..default()
    }
}

fn waterworks_emissive_material(base_color: Color, emissive_strength: f32) -> StandardMaterial {
    StandardMaterial {
        base_color,
        emissive: base_color.to_linear() * emissive_strength,
        unlit: true,
        perceptual_roughness: 0.8,
        reflectance: 0.0,
        ..default()
    }
}

fn waterworks_render_diagnostics_enabled() -> bool {
    std::env::var("SYMTROPY_OLD_WATERWORKS_RENDER_DIAGNOSTICS")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        || std::env::var("SYMTROPY_WATERWORKS_RENDER_DIAGNOSTICS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

fn waterworks_render_gate_camera_enabled() -> bool {
    std::env::var("SYMTROPY_DEMO_CAPTURE_DIR").is_ok()
        || std::env::var("SYMTROPY_OLD_WATERWORKS_GATE_CAMERA")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

fn waterworks_camera_offset() -> Vec3 {
    if waterworks_render_gate_camera_enabled() {
        // Stable top-down capture: sees authored markers and avoids wall occlusion across seeds.
        Vec3::new(0.0, 0.0, 260.0)
    } else {
        // Playable composition: closer to the player, still elevated enough for the current blockout.
        Vec3::new(0.0, -72.0, 46.0)
    }
}

fn waterworks_camera_up() -> Vec3 {
    if waterworks_render_gate_camera_enabled() {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

#[cfg(test)]
fn is_magenta_like(color: Color) -> bool {
    let srgba = color.to_srgba();
    srgba.red > 0.55 && srgba.blue > 0.55 && srgba.green < 0.35
}

pub fn setup_world_3d(
    mut commands: Commands,
    layout: Res<SiteLayout>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut physics_world: ResMut<PhysicsWorldRes>,
    old_cameras: Query<Entity, (With<Camera>, Without<Camera3D>)>,
    asset_server: Res<AssetServer>,
) {
    info!("Initializing Embodied 3D Layer: Old Waterworks");

    let old_concrete_color = asset_server.load("textures/waterworks/old_concrete_color.jpg");
    let old_concrete_normal = asset_server.load("textures/waterworks/old_concrete_normal.jpg");
    let wet_concrete_color = asset_server.load("textures/waterworks/wet_concrete_color.jpg");
    let wet_concrete_normal = asset_server.load("textures/waterworks/wet_concrete_normal.jpg");
    let painted_steel_color = asset_server.load("textures/waterworks/painted_steel_color.jpg");
    let painted_steel_normal = asset_server.load("textures/waterworks/painted_steel_normal.jpg");

    for entity in &old_cameras {
        commands.entity(entity).despawn();
    }

    // Spawn 3D Perspective Camera
    let camera_offset = waterworks_camera_offset();
    let player_start_3d = Vec3::new(layout.player_start.x, layout.player_start.y, 0.0);
    commands.spawn((
        Name::new("Old Waterworks 3D Camera"),
        CameraRenderGraph::new(Core3d),
        Camera3d::default(),
        NoIndirectDrawing,
        Transform::from_translation(player_start_3d + camera_offset).looking_at(
            Vec3::new(layout.player_start.x, layout.player_start.y, 8.0),
            waterworks_camera_up(),
        ),
        Camera3D,
    ));

    commands.spawn((
        Name::new("Old Waterworks interaction focus prompt"),
        Text2d::new(""),
        TextFont {
            font_size: 9.0,
            ..default()
        },
        TextColor(FIELD_DECK_CYAN),
        Transform::from_xyz(layout.player_start.x, layout.player_start.y, 26.0),
        Visibility::Hidden,
        InteractionFocusPrompt,
    ));

    // No AmbientLight entity here: in Bevy 0.18 it participates in the
    // camera/view component requirements and creates a raw Camera warning.
    // The Old Waterworks greybox materials are intentionally unlit.

    // Directional light representing overhead facility grids
    commands.spawn((
        DirectionalLight {
            illuminance: 1200.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 300.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let cols = layout.width as f32;
    let rows = layout.height as f32;

    // Real CC0 textures (ambientCG Concrete034/Concrete048) replacing the flat
    // greybox colors, shared by every wall/floor tile rather than one
    // material asset per tile.
    let wall_material = materials.add(waterworks_textured_material(
        old_concrete_color.clone(),
        old_concrete_normal.clone(),
        Color::WHITE,
        0.95,
    ));
    let floor_material = materials.add(waterworks_textured_material(
        wet_concrete_color.clone(),
        wet_concrete_normal.clone(),
        // Subtle dark-teal tint toward the original WET_CONCRETE mood
        // (real concrete texture alone reads too bright/dry for "wet").
        Color::srgb(0.55, 0.62, 0.65),
        0.6,
    ));

    // Build 3D mesh grid from layout tiles
    for (row_idx, row) in layout.tiles.iter().enumerate() {
        for (col_idx, &cell) in row.iter().enumerate() {
            let x = (col_idx as f32 - cols / 2.0) * TILE_SIZE;
            let y = (rows / 2.0 - row_idx as f32) * TILE_SIZE;

            if cell == 0 {
                // Wall: intentional old concrete, not renderer fallback magenta.
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(TILE_SIZE, TILE_SIZE, 60.0))),
                    MeshMaterial3d(wall_material.clone()),
                    Transform::from_xyz(x, y, 30.0),
                ));
            } else {
                // Floor: wet concrete placeholder material.
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(TILE_SIZE, TILE_SIZE, 2.0))),
                    MeshMaterial3d(floor_material.clone()),
                    Transform::from_xyz(x, y, -1.0),
                ));
            }
        }
    }

    // Spawn 3D Player entity (Cyan capsule)
    let player_physics_handle = physics_world.world.add_sphere(
        symtropy_math::Point::new([layout.player_start.x as f64, layout.player_start.y as f64]),
        PLAYER_PHYSICS_RADIUS,
        1.0,
    );
    if let Some(body) = physics_world.world.body_mut(player_physics_handle) {
        body.linear_damping = 0.5;
    }

    commands
        .spawn((
            Name::new("Old Waterworks Player"),
            Transform::from_xyz(layout.player_start.x, layout.player_start.y, 0.0),
            Visibility::Visible,
            InheritedVisibility::default(),
            Player,
            Player3D,
            Flashlight::default(),
            NoiseEmitter::default(),
            ConsciousnessComp::default(),
            TendBalance::new(40),
            FactionAffiliation::default(),
            symtropy_render_bridge::PhysicsBody::new(
                player_physics_handle,
                PLAYER_COLLISION_RADIUS,
            ),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(5.5, 3.0, PLAYER_VISUAL_HEIGHT))),
                MeshMaterial3d(materials.add(waterworks_textured_material(
                    painted_steel_color.clone(),
                    painted_steel_normal.clone(),
                    PAINTED_STEEL,
                    0.4,
                ))),
                Transform::from_xyz(0.0, 0.0, 4.0),
            ));
            parent.spawn((
                Mesh3d(meshes.add(Sphere::new(2.0).mesh().uv(12, 8))),
                MeshMaterial3d(materials.add(waterworks_material(OLD_CONCRETE))),
                Transform::from_xyz(0.0, 0.0, 9.0),
            ));
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(4.5, 1.0, 2.0))),
                MeshMaterial3d(materials.add(waterworks_emissive_material(FIELD_DECK_CYAN, 0.35))),
                Transform::from_xyz(0.0, -2.2, 5.2),
            ));
        });

    // Spawn 3D machines at room centers
    let room_centers = &layout.room_centers;
    if !room_centers.is_empty() {
        // 1. Power Junction (Room Center 1 or fallback)
        let idx_power = 1 % room_centers.len();
        let pos_power = room_centers[idx_power];
        let x_pow = (pos_power.0 as f32 - cols / 2.0) * TILE_SIZE;
        let y_pow = (rows / 2.0 - pos_power.1 as f32) * TILE_SIZE;
        commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(18.0, 18.0, 24.0))),
                MeshMaterial3d(materials.add(waterworks_emissive_material(WARNING_AMBER, 0.18))),
                Transform::from_xyz(x_pow, y_pow, 12.0),
                PowerJunction {
                    output: 0.2,
                    is_damaged: true,
                    uptime_secs: 0.0,
                },
                InteractionTarget {
                    radius: 40.0,
                    label: "Repair Power Junction".to_string(),
                },
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text2d::new("Power Junction (Offline)"),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.8, 0.6)),
                    Transform::from_xyz(0.0, 0.0, 20.0),
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(26.0, 4.0, 5.0))),
                    MeshMaterial3d(materials.add(waterworks_material(WARNING_AMBER))),
                    Transform::from_xyz(0.0, -13.0, 2.0),
                    JunctionStatusBar,
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(13.0, 2.0, 7.0))),
                    MeshMaterial3d(materials.add(waterworks_emissive_material(REPAIR_TAG, 0.25))),
                    Transform::from_xyz(0.0, -13.6, 9.0),
                    Visibility::Hidden,
                    JunctionEvidenceTag,
                ));
                parent.spawn((
                    Text2d::new("CHRONICLE SEALED"),
                    TextFont {
                        font_size: 5.5,
                        ..default()
                    },
                    TextColor(REPAIR_TAG),
                    Transform::from_xyz(0.0, -15.0, 14.0),
                    Visibility::Hidden,
                    JunctionEvidenceTag,
                ));
            });

        // 2. Water Pump (Room Center 2 or fallback)
        let idx_pump = if room_centers.len() > 2 { 2 } else { 0 };
        let pos_pump = room_centers[idx_pump];
        let x_pump = (pos_pump.0 as f32 - cols / 2.0) * TILE_SIZE;
        let y_pump = (rows / 2.0 - pos_pump.1 as f32) * TILE_SIZE;
        commands
            .spawn((
                Mesh3d(meshes.add(Cylinder::new(9.0, 24.0))),
                MeshMaterial3d(materials.add(waterworks_material(PAINTED_STEEL))),
                Transform::from_xyz(x_pump, y_pump, 12.0),
                WaterPump {
                    efficiency: 1.0,
                    is_running: false,
                    is_sabotaged: true,
                },
                InteractionTarget {
                    radius: 40.0,
                    label: "Restore Water Pump".to_string(),
                },
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text2d::new("Water Pump (Sabotaged)"),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.6, 0.8, 1.0)),
                    Transform::from_xyz(0.0, 0.0, 20.0),
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(44.0, 5.0, 5.0))),
                    MeshMaterial3d(materials.add(waterworks_material(RUSTED_METAL))),
                    Transform::from_xyz(-28.0, 0.0, 0.0),
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(44.0, 5.0, 5.0))),
                    MeshMaterial3d(materials.add(waterworks_material(RUSTED_METAL))),
                    Transform::from_xyz(28.0, 0.0, 0.0),
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(18.0, 7.0, 7.0))),
                    MeshMaterial3d(materials.add(waterworks_emissive_material(REPAIR_TAG, 0.22))),
                    Transform::from_xyz(0.0, 0.0, 0.0),
                    Visibility::Hidden,
                    PumpSealBand,
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(5.0, 2.0, 5.0))),
                    MeshMaterial3d(
                        materials.add(waterworks_emissive_material(FIELD_DECK_CYAN, 0.3)),
                    ),
                    Transform::from_xyz(0.0, -10.0, 8.0),
                    Visibility::Hidden,
                    PumpFlowLight,
                ));
            });

        // 3. Fabricator (Room Center 3 or fallback)
        let idx_fab = if room_centers.len() > 3 { 3 } else { 0 };
        let pos_fab = room_centers[idx_fab];
        let x_fab = (pos_fab.0 as f32 - cols / 2.0) * TILE_SIZE;
        let y_fab = (rows / 2.0 - pos_fab.1 as f32) * TILE_SIZE;
        commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(20.0, 20.0, 20.0))),
                MeshMaterial3d(materials.add(waterworks_material(RUSTED_METAL))),
                Transform::from_xyz(x_fab, y_fab, 10.0),
                Fabricator {
                    material_reserve: 0.5,
                    power_draw: 0.1,
                    is_active: false,
                },
                InteractionTarget {
                    radius: 40.0,
                    label: "Access Fabricator".to_string(),
                },
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text2d::new("Fabricator"),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.9, 0.7, 1.0)),
                    Transform::from_xyz(0.0, 0.0, 20.0),
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(18.0, 4.0, 3.0))),
                    MeshMaterial3d(materials.add(waterworks_emissive_material(REPAIR_TAG, 0.2))),
                    Transform::from_xyz(0.0, -12.0, 8.0),
                    FabricatorStatusPanel,
                ));
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(10.0, 2.0, 5.0))),
                    MeshMaterial3d(materials.add(waterworks_emissive_material(CORE_GOLD, 0.22))),
                    Transform::from_xyz(0.0, -12.8, 13.0),
                    Visibility::Hidden,
                    FabricatorEvidenceTag,
                ));
            });
    }

    spawn_waterworks_environment_props(&layout, &mut commands, &mut meshes, &mut materials);

    // Spawn 7 NPC archetypes in 3D
    let npc_configs = [
        (
            "Engineer (Kael)",
            layout.player_start.x - 44.0,
            layout.player_start.y,
            WORKWEAR_ORANGE,
            0.4, // caution
        ),
        (
            "Medic (Mira)",
            layout.player_start.x + 44.0,
            layout.player_start.y,
            MEDIC_GREEN,
            0.6, // caution
        ),
        (
            "Archivist (Soren)",
            layout.player_start.x,
            layout.player_start.y + 44.0,
            ARCHIVE_BLUE,
            0.7, // caution
        ),
        (
            "Convoy Lead (Jack)",
            layout.player_start.x - 44.0,
            layout.player_start.y - 44.0,
            CONVOY_RED,
            0.3, // caution
        ),
        (
            "Friendly Robot (PR-4)",
            layout.player_start.x + 44.0,
            layout.player_start.y - 44.0,
            ROBOT_TEAL,
            0.1, // caution
        ),
        (
            "Industrial Liaison (Nadia)",
            layout.player_start.x - 78.0,
            layout.player_start.y,
            LIAISON_OCHRE,
            0.5, // caution
        ),
        (
            "Young Tech (Leo)",
            layout.player_start.x + 78.0,
            layout.player_start.y,
            TECH_GREEN,
            0.3, // caution
        ),
    ];

    let npc_consciousness = [
        [0.6, 0.5, 0.5, 0.7, 0.6, 0.5], // Engineer: high care
        [0.4, 0.3, 0.6, 0.5, 0.4, 0.4], // Medic: community/social
        [0.8, 0.7, 0.7, 0.4, 0.8, 0.6], // Archivist: deep knowledge
        [0.5, 0.6, 0.4, 0.7, 0.5, 0.5], // Convoy Lead: tactical
        [0.2, 0.1, 0.8, 0.9, 0.2, 0.1], // Friendly Robot: low self-identity, high engagement
        [0.7, 0.8, 0.5, 0.5, 0.7, 0.6], // Industrial Liaison: negotiation
        [0.3, 0.4, 0.4, 0.6, 0.3, 0.3], // Young Tech: junior
    ];

    for (i, (name, x, y, color, caution)) in npc_configs.iter().enumerate() {
        let cp = ConsciousnessComp {
            sim_dimensions: npc_consciousness[i],
            ..Default::default()
        };

        // Register NPC physics body (2D physics world)
        let npc_physics_handle = physics_world.world.add_sphere(
            symtropy_math::Point::new([*x as f64, *y as f64]),
            NPC_COLLISION_RADIUS as f64,
            1.0,
        );
        if let Some(body) = physics_world.world.body_mut(npc_physics_handle) {
            body.linear_damping = 0.5;
        }
        physics_world
            .consciousness
            .register(npc_physics_handle, 80.0, 30.0);

        let mut crew_npc = CrewNpc::new(name, i as u64 + 100);
        crew_npc.caution = *caution;

        let mut entity_cmds = commands.spawn((
            Name::new(format!("Old Waterworks {name}")),
            Transform::from_xyz(*x, *y, 0.0),
            Visibility::Visible,
            InheritedVisibility::default(),
            crew_npc,
            MoveTarget {
                target: None,
                speed: 60.0,
            },
            NoiseEmitter::default(),
            cp,
            NpcConsciousness::default(),
            PsychologicalNeeds {
                allostatic_load: match i {
                    0 => 0.20, // Kael
                    1 => 0.40, // Mira
                    2 => 0.10, // Soren
                    3 => 0.30, // Jack
                    4 => 0.05, // PR-4
                    5 => 0.25, // Nadia
                    _ => 0.50, // Leo
                },
                social_satiation: match i {
                    0 => 0.40,
                    1 => 0.80,
                    2 => 0.30,
                    3 => 0.50,
                    4 => 0.90,
                    5 => 0.60,
                    _ => 0.50,
                },
                engagement: match i {
                    0 => 0.90,
                    1 => 0.70,
                    2 => 0.80,
                    3 => 0.60,
                    4 => 0.95,
                    5 => 0.75,
                    _ => 0.80,
                },
                burnout_ticks: 0,
            },
            TendBalance::new(40),
            FactionAffiliation {
                faction_id: None,
                ideology: [
                    npc_consciousness[i][0],
                    npc_consciousness[i][1],
                    npc_consciousness[i][3],
                    npc_consciousness[i][4],
                ],
            },
            NpcTrust::default(),
            symtropy_render_bridge::PhysicsBody::new(npc_physics_handle, NPC_COLLISION_RADIUS),
        ));
        entity_cmds.with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(5.0, 3.0, 7.0))),
                MeshMaterial3d(materials.add(waterworks_material(*color))),
                Transform::from_xyz(0.0, 0.0, 3.7),
            ));
            parent.spawn((
                Mesh3d(meshes.add(Sphere::new(1.8).mesh().uv(12, 8))),
                MeshMaterial3d(materials.add(waterworks_material(OLD_CONCRETE))),
                Transform::from_xyz(0.0, 0.0, 8.4),
            ));
        });

        #[cfg(feature = "symthaea-bevy-brain")]
        if name.contains("Mira") {
            entity_cmds.insert(symthaea_bevy_brain::CognitiveBrain::new_with_hv_input(
                64,
                "mira_medic_genesis",
                16_384,
            ));
        }
    }

    // Spawn 3D Fusion Core
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(10.0).mesh().uv(16, 16))),
        MeshMaterial3d(materials.add(waterworks_emissive_material(CORE_GOLD, 0.45))),
        Transform::from_xyz(layout.core_pos.x, layout.core_pos.y, 10.0),
        FusionCore {
            being_extracted: false,
            extraction_progress: 0.0,
        },
    ));

    // Spawn 3D Leviathan
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(32.0, 32.0, 32.0))),
        MeshMaterial3d(materials.add(waterworks_emissive_material(LEVIATHAN_RED, 0.15))),
        Transform::from_xyz(layout.core_pos.x, layout.core_pos.y + 64.0, 16.0),
        LeviathanSprite,
    ));

    // Spawn light point at core room
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.8, 0.2),
            intensity: 8000.0,
            range: 150.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(layout.core_pos.x, layout.core_pos.y, 25.0),
    ));

    if waterworks_render_diagnostics_enabled() {
        spawn_waterworks_render_diagnostics(&mut commands, &mut meshes, &mut materials);
    }
}

fn spawn_waterworks_environment_props(
    layout: &SiteLayout,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let pipe_mat = materials.add(waterworks_material(RUSTED_METAL));
    let water_mat = materials.add(waterworks_material(DIRTY_WATER));
    let stain_mat = materials.add(waterworks_material(ALGAE_STAIN));
    let amber_mat = materials.add(waterworks_emissive_material(WARNING_AMBER, 0.12));
    let tag_mat = materials.add(waterworks_emissive_material(REPAIR_TAG, 0.18));
    let field_deck_mat = materials.add(waterworks_material(FIELD_DECK_CYAN));
    let steel_mat = materials.add(waterworks_material(PAINTED_STEEL));
    let rubber_mat = materials.add(waterworks_material(RUBBER_BLACK));

    let origin = layout.player_start;
    let core = layout.core_pos;

    // Long service pipes create municipal infrastructure silhouettes and scale cues.
    for (idx, y_offset) in [-48.0, 48.0].iter().enumerate() {
        commands.spawn((
            Name::new(format!("Old Waterworks service pipe {}", idx + 1)),
            Mesh3d(meshes.add(Cuboid::new(220.0, 5.0, 5.0))),
            MeshMaterial3d(pipe_mat.clone()),
            Transform::from_xyz(origin.x + 40.0, origin.y + y_offset, 18.0),
        ));
    }

    // Floodwater and algae stains read as deliberate environmental state, not a flat test floor.
    commands.spawn((
        Name::new("Old Waterworks shallow floodwater"),
        Mesh3d(meshes.add(Cuboid::new(90.0, 54.0, 0.7))),
        MeshMaterial3d(water_mat),
        Transform::from_xyz(origin.x + 70.0, origin.y - 38.0, 0.4),
    ));
    commands.spawn((
        Name::new("Old Waterworks algae waterline stain"),
        Mesh3d(meshes.add(Cuboid::new(150.0, 4.0, 8.0))),
        MeshMaterial3d(stain_mat),
        Transform::from_xyz(origin.x + 28.0, origin.y + 63.0, 5.0),
    ));

    // Warning strip and evidence tag make the repair/Chronicle layer visible in-world.
    commands.spawn((
        Name::new("Old Waterworks warning floor stripe"),
        Mesh3d(meshes.add(Cuboid::new(120.0, 3.0, 1.0))),
        MeshMaterial3d(amber_mat.clone()),
        Transform::from_xyz(origin.x + 20.0, origin.y - 18.0, 1.2),
    ));
    commands.spawn((
        Name::new("Old Waterworks Chronicle evidence tag"),
        Mesh3d(meshes.add(Cuboid::new(18.0, 2.0, 12.0))),
        MeshMaterial3d(tag_mat.clone()),
        Transform::from_xyz(core.x + 18.0, core.y, 12.0),
    ));
    commands.spawn((
        Name::new("Old Waterworks Field Deck scan marker"),
        Mesh3d(meshes.add(Cuboid::new(34.0, 16.0, 1.0))),
        MeshMaterial3d(field_deck_mat),
        Transform::from_xyz(origin.x - 28.0, origin.y - 34.0, 2.0),
    ));

    // Primitive prop kit: cheap silhouettes that establish waterworks identity.
    spawn_grated_walkway(
        commands,
        meshes,
        steel_mat.clone(),
        rubber_mat.clone(),
        Vec3::new(origin.x + 12.0, origin.y + 22.0, 2.0),
    );
    spawn_valve_wheel(
        commands,
        meshes,
        pipe_mat.clone(),
        amber_mat.clone(),
        Vec3::new(origin.x - 52.0, origin.y + 48.0, 20.0),
    );
    spawn_control_cabinet(
        commands,
        meshes,
        steel_mat.clone(),
        amber_mat.clone(),
        Vec3::new(origin.x - 72.0, origin.y - 24.0, 13.0),
    );
    spawn_pressure_gauge(
        commands,
        meshes,
        steel_mat,
        tag_mat.clone(),
        Vec3::new(origin.x + 76.0, origin.y + 48.0, 16.0),
    );
    spawn_repair_crate(
        commands,
        meshes,
        materials.add(waterworks_material(OLD_CONCRETE)),
        Vec3::new(origin.x + 64.0, origin.y + 12.0, 5.0),
    );
}

fn spawn_grated_walkway(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    steel_mat: Handle<StandardMaterial>,
    rubber_mat: Handle<StandardMaterial>,
    origin: Vec3,
) {
    commands.spawn((
        Name::new("Old Waterworks grated walkway base"),
        Mesh3d(meshes.add(Cuboid::new(86.0, 22.0, 1.0))),
        MeshMaterial3d(rubber_mat),
        Transform::from_translation(origin),
    ));
    for idx in 0..8 {
        let x = origin.x - 35.0 + idx as f32 * 10.0;
        commands.spawn((
            Name::new("Old Waterworks grate slat"),
            Mesh3d(meshes.add(Cuboid::new(2.0, 22.0, 1.4))),
            MeshMaterial3d(steel_mat.clone()),
            Transform::from_xyz(x, origin.y, origin.z + 0.8),
        ));
    }
}

fn spawn_valve_wheel(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    pipe_mat: Handle<StandardMaterial>,
    marker_mat: Handle<StandardMaterial>,
    origin: Vec3,
) {
    commands.spawn((
        Name::new("Old Waterworks valve hub"),
        Mesh3d(meshes.add(Cylinder::new(4.0, 3.0))),
        MeshMaterial3d(pipe_mat.clone()),
        Transform::from_translation(origin),
    ));
    for rotation in [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        std::f32::consts::PI * 1.5,
    ] {
        commands.spawn((
            Name::new("Old Waterworks valve spoke"),
            Mesh3d(meshes.add(Cuboid::new(20.0, 2.0, 2.0))),
            MeshMaterial3d(pipe_mat.clone()),
            Transform {
                translation: origin,
                rotation: Quat::from_rotation_z(rotation),
                ..default()
            },
        ));
    }
    commands.spawn((
        Name::new("Old Waterworks valve marker"),
        Mesh3d(meshes.add(Cuboid::new(5.0, 2.0, 2.5))),
        MeshMaterial3d(marker_mat),
        Transform::from_xyz(origin.x + 10.0, origin.y, origin.z + 1.0),
    ));
}

fn spawn_control_cabinet(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    steel_mat: Handle<StandardMaterial>,
    light_mat: Handle<StandardMaterial>,
    origin: Vec3,
) {
    commands.spawn((
        Name::new("Old Waterworks control cabinet"),
        Mesh3d(meshes.add(Cuboid::new(20.0, 8.0, 26.0))),
        MeshMaterial3d(steel_mat),
        Transform::from_translation(origin),
    ));
    commands.spawn((
        Name::new("Old Waterworks cabinet status light"),
        Mesh3d(meshes.add(Cuboid::new(7.0, 1.5, 4.0))),
        MeshMaterial3d(light_mat),
        Transform::from_xyz(origin.x, origin.y - 4.8, origin.z + 7.0),
    ));
}

fn spawn_pressure_gauge(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    steel_mat: Handle<StandardMaterial>,
    tag_mat: Handle<StandardMaterial>,
    origin: Vec3,
) {
    commands.spawn((
        Name::new("Old Waterworks pressure gauge body"),
        Mesh3d(meshes.add(Cylinder::new(6.0, 2.0))),
        MeshMaterial3d(steel_mat),
        Transform::from_translation(origin),
    ));
    commands.spawn((
        Name::new("Old Waterworks pressure gauge needle"),
        Mesh3d(meshes.add(Cuboid::new(8.0, 1.0, 1.0))),
        MeshMaterial3d(tag_mat),
        Transform {
            translation: Vec3::new(origin.x + 1.5, origin.y, origin.z + 1.2),
            rotation: Quat::from_rotation_z(0.45),
            ..default()
        },
    ));
}

fn spawn_repair_crate(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
    origin: Vec3,
) {
    commands.spawn((
        Name::new("Old Waterworks repair crate"),
        Mesh3d(meshes.add(Cuboid::new(16.0, 12.0, 10.0))),
        MeshMaterial3d(material),
        Transform::from_translation(origin),
    ));
}

fn spawn_waterworks_render_diagnostics(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let swatches = [
        ("wet concrete", WET_CONCRETE),
        ("old concrete", OLD_CONCRETE),
        ("rusted metal", RUSTED_METAL),
        ("painted steel", PAINTED_STEEL),
        ("warning amber", WARNING_AMBER),
        ("field deck", FIELD_DECK_CYAN),
    ];

    let origin = Vec3::new(-220.0, -180.0, 18.0);
    for (idx, (label, color)) in swatches.iter().enumerate() {
        let x = origin.x + idx as f32 * 26.0;
        commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(18.0, 18.0, 18.0))),
                MeshMaterial3d(materials.add(waterworks_material(*color))),
                Transform::from_xyz(x, origin.y, origin.z),
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text2d::new(*label),
                    TextFont {
                        font_size: 6.0,
                        ..default()
                    },
                    TextColor(Color::WHITE),
                    Transform::from_xyz(0.0, -16.0, 14.0),
                ));
            });
    }
}

pub fn cleanup_non_3d_cameras(
    mut commands: Commands,
    old_cameras: Query<
        (
            Entity,
            Option<&Name>,
            Has<Camera2d>,
            Has<Camera3d>,
            Has<CameraRenderGraph>,
        ),
        (With<Camera>, Without<Camera3D>),
    >,
) {
    for (entity, name, has_camera_2d, has_camera_3d, has_render_graph) in &old_cameras {
        if waterworks_render_diagnostics_enabled() {
            info!(
                "Old Waterworks removing non-3D camera: entity={entity:?}, name={}, camera2d={has_camera_2d}, camera3d={has_camera_3d}, render_graph={has_render_graph}",
                name.map(Name::as_str).unwrap_or("<unnamed>")
            );
        }
        commands.entity(entity).despawn();
    }
}

pub fn waterworks_render_diagnostics_system(
    cameras: Query<
        (
            Entity,
            Option<&Name>,
            Has<Camera3d>,
            Has<CameraRenderGraph>,
            Has<Camera3D>,
        ),
        With<Camera>,
    >,
    mut logged: Local<bool>,
) {
    if *logged || !waterworks_render_diagnostics_enabled() {
        return;
    }

    *logged = true;
    let mut active_3d_count = 0;
    let mut raw_camera_count = 0;
    for (entity, name, has_camera_3d, has_render_graph, is_waterworks_camera) in &cameras {
        if has_camera_3d && has_render_graph {
            active_3d_count += 1;
        }
        if !has_camera_3d && !has_render_graph {
            raw_camera_count += 1;
        }
        info!(
            "Old Waterworks camera diagnostic: entity={entity:?}, name={}, camera3d={has_camera_3d}, render_graph={has_render_graph}, waterworks_camera={is_waterworks_camera}",
            name.map(Name::as_str).unwrap_or("<unnamed>")
        );
    }
    if active_3d_count != 1 || raw_camera_count > 0 {
        warn!(
            "Old Waterworks camera gate: active_3d_count={active_3d_count}, raw_camera_count={raw_camera_count}"
        );
    } else {
        info!("Old Waterworks camera gate: active_3d_count=1, raw_camera_count=0");
    }
}

pub fn waterworks_raw_camera_component_diagnostics_system(world: &mut World) {
    if !waterworks_render_diagnostics_enabled() {
        return;
    }

    let mut query = world.query_filtered::<Entity, (With<Camera>, Without<Camera3D>)>();
    let raw_cameras: Vec<Entity> = query.iter(world).collect();
    for entity in raw_cameras {
        let Ok(component_info) = world.inspect_entity(entity) else {
            continue;
        };
        let components = component_info
            .map(|info| info.name().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        info!("Old Waterworks raw camera components: entity={entity:?}, components=[{components}]");
    }
}

fn strict_walkable_at(tile_grid: &TileGrid, world_x: f32, world_y: f32) -> bool {
    if tile_grid.tile_size <= 0.0 {
        return false;
    }

    let col = ((world_x / tile_grid.tile_size) + tile_grid.origin_col as f32 + 0.5).floor() as i32;
    let row = (tile_grid.origin_row as f32 - (world_y / tile_grid.tile_size) + 0.5).floor() as i32;

    if col < 0 || row < 0 || col >= tile_grid.cols || row >= tile_grid.rows {
        return false;
    }

    tile_grid.cells.get(&(col, row)).copied().unwrap_or(false)
}

fn can_occupy_position(tile_grid: &TileGrid, pos: Vec2, radius: f32) -> bool {
    let half_radius = radius * 0.5;
    let samples = [
        pos,
        pos + Vec2::new(radius, 0.0),
        pos + Vec2::new(-radius, 0.0),
        pos + Vec2::new(0.0, radius),
        pos + Vec2::new(0.0, -radius),
        pos + Vec2::new(half_radius, half_radius),
        pos + Vec2::new(half_radius, -half_radius),
        pos + Vec2::new(-half_radius, half_radius),
        pos + Vec2::new(-half_radius, -half_radius),
        pos + Vec2::new(radius, radius),
        pos + Vec2::new(radius, -radius),
        pos + Vec2::new(-radius, radius),
        pos + Vec2::new(-radius, -radius),
    ];

    samples
        .into_iter()
        .all(|sample| strict_walkable_at(tile_grid, sample.x, sample.y))
}

/// Simple 3D player controller matching 2D controls but operating on 3D physics transforms.
pub fn player_movement_system_3d(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &symtropy_render_bridge::PhysicsBody), With<Player3D>>,
    mut physics_world: ResMut<PhysicsWorldRes>,
    tile_grid: Res<TileGrid>,
    time: Res<Time>,
) {
    let Ok((mut transform, body_ref)) = query.single_mut() else {
        return;
    };
    let mut direction = Vec2::ZERO;

    if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) {
        direction.y += 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) {
        direction.y -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyA) || keyboard_input.pressed(KeyCode::ArrowLeft) {
        direction.x -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyD) || keyboard_input.pressed(KeyCode::ArrowRight) {
        direction.x += 1.0;
    }

    if direction != Vec2::ZERO {
        let delta_pos = direction.normalize() * PLAYER_MOVE_SPEED * time.delta_secs();
        let current_pos = transform.translation.truncate();
        let mut next_pos = current_pos;

        let candidate_x = Vec2::new(current_pos.x + delta_pos.x, current_pos.y);
        if can_occupy_position(&tile_grid, candidate_x, PLAYER_COLLISION_RADIUS) {
            next_pos.x = candidate_x.x;
        }

        let candidate_y = Vec2::new(next_pos.x, current_pos.y + delta_pos.y);
        if can_occupy_position(&tile_grid, candidate_y, PLAYER_COLLISION_RADIUS) {
            next_pos.y = candidate_y.y;
        }

        // Update physics representation
        if let Some(body) = physics_world.world.body_mut(body_ref.handle) {
            body.transform.translation =
                symtropy_math::Point::new([next_pos.x as f64, next_pos.y as f64]);

            // Sync player translation to the visual Transform immediately
            transform.translation.x = next_pos.x;
            transform.translation.y = next_pos.y;
        }

        // Apply visual rotation to face movement direction
        let angle = direction.y.atan2(direction.x) - std::f32::consts::FRAC_PI_2;
        transform.rotation = Quat::from_rotation_z(angle);
    }
}

fn interaction_prompt_text(
    label: &str,
    junction: Option<&PowerJunction>,
    pump: Option<&WaterPump>,
    fabricator: Option<&Fabricator>,
) -> String {
    if let Some(junction) = junction {
        if junction.is_damaged {
            return format!("[E] {label}");
        }
        return "Power Junction stable".to_string();
    }
    if let Some(pump) = pump {
        if pump.is_sabotaged || !pump.is_running {
            return format!("[E] {label}");
        }
        return "Water Pump running".to_string();
    }
    if let Some(fabricator) = fabricator {
        if !fabricator.is_active {
            return format!("[E] {label}");
        }
        return "Fabricator active".to_string();
    }
    format!("[E] {label}")
}

/// Updates the floating prompt for the nearest 3D interaction target.
pub fn interaction_focus_prompt_system_3d(
    player_query: Query<&Transform, (With<Player3D>, Without<InteractionFocusPrompt>)>,
    targets: Query<
        (
            &Transform,
            &InteractionTarget,
            Option<&PowerJunction>,
            Option<&WaterPump>,
            Option<&Fabricator>,
        ),
        Without<InteractionFocusPrompt>,
    >,
    mut prompt_query: Query<
        (&mut Text2d, &mut TextColor, &mut Transform, &mut Visibility),
        With<InteractionFocusPrompt>,
    >,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok((mut prompt, mut color, mut prompt_tf, mut visibility)) = prompt_query.single_mut()
    else {
        return;
    };

    let player_pos = player_tf.translation.truncate();
    let mut nearest: Option<(f32, Vec3, String)> = None;

    for (target_tf, target, junction, pump, fabricator) in &targets {
        let dist = player_pos.distance(target_tf.translation.truncate());
        if dist > target.radius {
            continue;
        }

        let label = interaction_prompt_text(&target.label, junction, pump, fabricator);
        if nearest
            .as_ref()
            .map(|(nearest_dist, _, _)| dist < *nearest_dist)
            .unwrap_or(true)
        {
            nearest = Some((dist, target_tf.translation, label));
        }
    }

    if let Some((_, target_pos, label)) = nearest {
        *prompt = Text2d::new(label.clone());
        *color = if label.starts_with("[E]") {
            TextColor(FIELD_DECK_CYAN)
        } else {
            TextColor(REPAIR_TAG)
        };
        prompt_tf.translation = target_pos + Vec3::new(0.0, -18.0, 30.0);
        *visibility = Visibility::Visible;
    } else {
        *prompt = Text2d::new("");
        *visibility = Visibility::Hidden;
    }
}

/// Handles embodied 3D interaction with repair targets.
pub fn interaction_system_3d(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<Player3D>>,
    mut junctions: Query<(&mut PowerJunction, &Transform, &InteractionTarget)>,
    mut pumps: Query<(&mut WaterPump, &Transform, &InteractionTarget)>,
    mut fabricators: Query<(&mut Fabricator, &Transform, &InteractionTarget)>,
    mut feedback_writer: MessageWriter<WorldFeedbackEvent>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) {
        return;
    }
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    let mut best_action: Option<(f32, Vec2, &'static str, Color)> = None;

    for (mut junction, tf, target) in &mut junctions {
        let dist = player_pos.distance(tf.translation.truncate());
        if dist <= target.radius && junction.is_damaged {
            junction.is_damaged = false;
            junction.output = 1.0;
            best_action = Some((
                dist,
                tf.translation.truncate(),
                "Power junction repaired",
                WARNING_AMBER,
            ));
        }
    }

    for (mut pump, tf, target) in &mut pumps {
        let dist = player_pos.distance(tf.translation.truncate());
        if dist <= target.radius && (pump.is_sabotaged || !pump.is_running) {
            pump.is_sabotaged = false;
            pump.is_running = true;
            if best_action
                .as_ref()
                .map(|(best_dist, _, _, _)| dist < *best_dist)
                .unwrap_or(true)
            {
                best_action = Some((
                    dist,
                    tf.translation.truncate(),
                    "Water pump restored",
                    FIELD_DECK_CYAN,
                ));
            }
        }
    }

    for (mut fabricator, tf, target) in &mut fabricators {
        let dist = player_pos.distance(tf.translation.truncate());
        if dist <= target.radius && !fabricator.is_active {
            fabricator.is_active = true;
            if best_action
                .as_ref()
                .map(|(best_dist, _, _, _)| dist < *best_dist)
                .unwrap_or(true)
            {
                best_action = Some((
                    dist,
                    tf.translation.truncate(),
                    "Fabricator brought online",
                    REPAIR_TAG,
                ));
            }
        }
    }

    if let Some((_, position, message, color)) = best_action {
        feedback_writer.write(WorldFeedbackEvent {
            position,
            message: message.to_string(),
            color,
        });
    }
}

pub fn machine_state_visual_system_3d(
    junctions: Query<
        (&PowerJunction, &MeshMaterial3d<StandardMaterial>, &Children),
        Changed<PowerJunction>,
    >,
    pumps: Query<(&WaterPump, &MeshMaterial3d<StandardMaterial>, &Children), Changed<WaterPump>>,
    fabricators: Query<
        (&Fabricator, &MeshMaterial3d<StandardMaterial>, &Children),
        Changed<Fabricator>,
    >,
    mut child_visuals: Query<(
        Option<&JunctionStatusBar>,
        Option<&JunctionEvidenceTag>,
        Option<&PumpSealBand>,
        Option<&PumpFlowLight>,
        Option<&FabricatorStatusPanel>,
        Option<&FabricatorEvidenceTag>,
        &mut Visibility,
        Option<&MeshMaterial3d<StandardMaterial>>,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (junction, material_handle, children) in &junctions {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            *material = if junction.is_damaged {
                waterworks_emissive_material(WARNING_AMBER, 0.18)
            } else {
                waterworks_emissive_material(REPAIR_TAG, 0.35)
            };
        }

        for child in children.iter() {
            let Ok((status_bar, evidence_tag, _, _, _, _, mut visibility, child_material)) =
                child_visuals.get_mut(child)
            else {
                continue;
            };

            if status_bar.is_some() {
                *visibility = Visibility::Visible;
                if let Some(child_material) = child_material {
                    if let Some(material) = materials.get_mut(&child_material.0) {
                        *material = if junction.is_damaged {
                            waterworks_material(WARNING_AMBER)
                        } else {
                            waterworks_emissive_material(REPAIR_TAG, 0.25)
                        };
                    }
                }
            }

            if evidence_tag.is_some() {
                *visibility = if junction.is_damaged {
                    Visibility::Hidden
                } else {
                    Visibility::Visible
                };
            }
        }
    }

    for (pump, material_handle, children) in &pumps {
        let restored = !pump.is_sabotaged && pump.is_running;
        if let Some(material) = materials.get_mut(&material_handle.0) {
            *material = if !restored {
                waterworks_material(PAINTED_STEEL)
            } else {
                waterworks_emissive_material(FIELD_DECK_CYAN, 0.22)
            };
        }

        for child in children.iter() {
            let Ok((_, _, seal_band, flow_light, _, _, mut visibility, _)) =
                child_visuals.get_mut(child)
            else {
                continue;
            };

            if seal_band.is_some() || flow_light.is_some() {
                *visibility = if restored {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }

    for (fabricator, material_handle, children) in &fabricators {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            *material = if fabricator.is_active {
                waterworks_emissive_material(REPAIR_TAG, 0.28)
            } else {
                waterworks_material(RUSTED_METAL)
            };
        }

        for child in children.iter() {
            let Ok((_, _, _, _, status_panel, evidence_tag, mut visibility, child_material)) =
                child_visuals.get_mut(child)
            else {
                continue;
            };

            if status_panel.is_some() {
                *visibility = Visibility::Visible;
                if let Some(child_material) = child_material {
                    if let Some(material) = materials.get_mut(&child_material.0) {
                        *material = if fabricator.is_active {
                            waterworks_emissive_material(FIELD_DECK_CYAN, 0.32)
                        } else {
                            waterworks_emissive_material(REPAIR_TAG, 0.2)
                        };
                    }
                }
            }

            if evidence_tag.is_some() {
                *visibility = if fabricator.is_active {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
}

/// Perspective 3D camera follow system.
pub fn camera_follow_system_3d(
    player_query: Query<&Transform, (With<Player3D>, Without<Camera3D>)>,
    mut camera_query: Query<&mut Transform, (With<Camera3D>, Without<Player3D>)>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok(mut camera_tf) = camera_query.single_mut() else {
        return;
    };

    let target_pos = player_tf.translation + waterworks_camera_offset();
    camera_tf.translation = camera_tf
        .translation
        .lerp(target_pos, 8.0 * time.delta_secs());

    camera_tf.look_at(
        player_tf.translation + Vec3::new(0.0, 0.0, 8.0),
        waterworks_camera_up(),
    );
}

/// Update 3D Leviathan material visibility and pulsing based on phase.
pub fn leviathan_visual_system_3d(
    leviathan: Res<LeviathanState>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>, With<LeviathanSprite>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
) {
    let alpha = match leviathan.phase {
        SleepPhase::Dormant => 0.08,
        SleepPhase::Stirring => 0.3 + (time.elapsed_secs() * 2.0).sin().abs() * 0.2,
        SleepPhase::Awake => 0.7,
        SleepPhase::Hunting => 1.0,
    };
    for mat_handle in &materials_query {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let red = (0.18 + alpha * 0.72).clamp(0.0, 1.0);
            mat.base_color = Color::srgb(red, 0.08, 0.06);
            mat.emissive = mat.base_color.to_linear() * (0.05 + alpha * 0.35);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_waterworks_materials_are_intentional_not_magenta_fallbacks() {
        let palette = [
            WET_CONCRETE,
            OLD_CONCRETE,
            RUSTED_METAL,
            PAINTED_STEEL,
            WARNING_AMBER,
            FIELD_DECK_CYAN,
            CORE_GOLD,
        ];

        for color in palette {
            assert!(
                !is_magenta_like(color),
                "Old Waterworks palette must not include magenta-like fallback colors: {color:?}"
            );
        }
    }

    #[test]
    fn placeholder_materials_are_unlit_and_texture_independent() {
        let material = waterworks_material(WET_CONCRETE);
        assert!(material.unlit);
        assert_eq!(material.base_color, WET_CONCRETE);
        assert!(material.base_color_texture.is_none());
    }

    #[test]
    fn player_occupancy_respects_tile_grid_walls() {
        let mut grid = TileGrid {
            tile_size: 32.0,
            origin_col: 1,
            origin_row: 1,
            cols: 3,
            rows: 3,
            ..default()
        };
        for row in 0..3 {
            for col in 0..3 {
                grid.cells.insert((col, row), true);
            }
        }
        grid.cells.insert((2, 1), false);

        assert!(can_occupy_position(&grid, Vec2::ZERO, 6.0));
        assert!(!can_occupy_position(&grid, Vec2::new(31.0, 0.0), 6.0));
    }

    #[test]
    fn strict_walkable_uses_tile_bounds_not_rounding() {
        let mut grid = TileGrid {
            tile_size: 32.0,
            origin_col: 1,
            origin_row: 1,
            cols: 3,
            rows: 3,
            ..default()
        };
        grid.cells.insert((1, 1), true);
        grid.cells.insert((2, 1), false);

        assert!(strict_walkable_at(&grid, 15.9, 0.0));
        assert!(!strict_walkable_at(&grid, 16.1, 0.0));
        assert!(!strict_walkable_at(&grid, 999.0, 999.0));
    }
}
