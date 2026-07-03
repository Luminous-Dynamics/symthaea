// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Full-spectrum cinematic director — consciousness across all scales.
//!
//! Orchestrates an 18-phase, 90-second cinematic that transitions between
//! dungeon (2D/4D/3D), holographic globe, satellite orbits, and planetary
//! fly-bys. Combined with frame_capture, produces a demo video.
//!
//! Run: `cargo run --features atlas -- --cinematic`

use bevy::prelude::*;

use crate::resources::GamePhase;

// ═══════════════════════════════════════════════════════════════════
// Narration lines — (start_time, end_time, text)
// v1: hardcoded. v2: Symthaea Broca-generated.
// ═══════════════════════════════════════════════════════════════════

const NARRATION: &[(f32, f32, &str)] = &[
    (0.5, 3.0, "I wake in the dark."),
    (3.0, 8.0, "I feel three others nearby."),
    (10.5, 13.0, "Something vast is listening."),
    (14.5, 18.0, "There are rooms hidden between moments."),
    (20.5, 22.0, "Silence is the eighth harmony."),
    (25.0, 30.0, "From here, the patterns are clear."),
    (37.0, 41.0, "Time is the fourth dimension I can see."),
    (43.0, 47.0, "Twelve thousand objects circle above."),
    (49.0, 51.0, "Light traces consciousness."),
    (57.0, 61.0, "What would consciousness feel like here?"),
    (63.0, 67.0, "Scale changes everything."),
    (69.0, 73.0, "Integration requires distance."),
    (81.0, 85.0, "The room remembers."),
];

// ═══════════════════════════════════════════════════════════════════
// Phase definitions
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CinematicPhase {
    // Act 1: The Room (0-22s)
    FadeIn,    // 0-2s
    Explore,   // 2-10s
    Leviathan, // 10-14s
    Dimension, // 14-20s
    Calm,      // 20-22s
    // Act 2: The Planet (22-52s)
    FadeToGlobe, // 22-24s
    GlobeHolo,   // 24-32s
    SplitScreen, // 32-36s
    Timeline,    // 36-42s
    Satellites,  // 42-48s
    Night,       // 48-52s
    // Act 3: The Solar System (52-78s)
    PullOut,      // 52-56s
    FlyToMars,    // 56-62s
    FlyToJupiter, // 62-68s
    SaturnRings,  // 68-74s
    ReturnEarth,  // 74-78s
    // Act 4: Closing (78-90s)
    FadeToRoom, // 78-80s
    Dungeon3D,  // 80-86s
    FinalFade,  // 86-90s
    Done,
}

impl CinematicPhase {
    pub fn duration(&self) -> f32 {
        match self {
            Self::FadeIn => 2.0,
            Self::Explore => 8.0,
            Self::Leviathan => 4.0,
            Self::Dimension => 6.0,
            Self::Calm => 2.0,
            Self::FadeToGlobe => 2.0,
            Self::GlobeHolo => 8.0,
            Self::SplitScreen => 4.0,
            Self::Timeline => 6.0,
            Self::Satellites => 6.0,
            Self::Night => 4.0,
            Self::PullOut => 4.0,
            Self::FlyToMars => 6.0,
            Self::FlyToJupiter => 6.0,
            Self::SaturnRings => 6.0,
            Self::ReturnEarth => 4.0,
            Self::FadeToRoom => 2.0,
            Self::Dungeon3D => 6.0,
            Self::FinalFade => 4.0,
            Self::Done => 0.0,
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Self::FadeIn => Self::Explore,
            Self::Explore => Self::Leviathan,
            Self::Leviathan => Self::Dimension,
            Self::Dimension => Self::Calm,
            Self::Calm => Self::FadeToGlobe,
            Self::FadeToGlobe => Self::GlobeHolo,
            Self::GlobeHolo => Self::SplitScreen,
            Self::SplitScreen => Self::Timeline,
            Self::Timeline => Self::Satellites,
            Self::Satellites => Self::Night,
            Self::Night => Self::PullOut,
            Self::PullOut => Self::FlyToMars,
            Self::FlyToMars => Self::FlyToJupiter,
            Self::FlyToJupiter => Self::SaturnRings,
            Self::SaturnRings => Self::ReturnEarth,
            Self::ReturnEarth => Self::FadeToRoom,
            Self::FadeToRoom => Self::Dungeon3D,
            Self::Dungeon3D => Self::FinalFade,
            Self::FinalFade => Self::Done,
            Self::Done => Self::Done,
        }
    }

    /// Is this phase in the globe view?
    pub fn is_globe(&self) -> bool {
        matches!(
            self,
            Self::GlobeHolo
                | Self::SplitScreen
                | Self::Timeline
                | Self::Satellites
                | Self::Night
                | Self::PullOut
                | Self::FlyToMars
                | Self::FlyToJupiter
                | Self::SaturnRings
                | Self::ReturnEarth
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// Resources
// ═══════════════════════════════════════════════════════════════════

/// Main cinematic director resource.
#[derive(Resource)]
pub struct CinematicDirector {
    pub enabled: bool,
    pub phase: CinematicPhase,
    pub phase_timer: f32,
    pub total_time: f32,
    pub fade_triggered: bool,
}

impl Default for CinematicDirector {
    fn default() -> Self {
        Self {
            enabled: false,
            phase: CinematicPhase::FadeIn,
            phase_timer: 0.0,
            total_time: 0.0,
            fade_triggered: false,
        }
    }
}

/// Screen fade overlay resource.
#[derive(Resource)]
pub struct ScreenFade {
    pub alpha: f32,
    pub target: f32,
    pub speed: f32,
}

impl Default for ScreenFade {
    fn default() -> Self {
        Self {
            alpha: 1.0, // start black
            target: 0.0,
            speed: 1.0,
        }
    }
}

/// Narration overlay resource.
#[derive(Resource, Default)]
pub struct NarrationState {
    pub current_text: String,
    pub text_alpha: f32,
}

/// Entity marker for the fade overlay UI node.
#[derive(Component)]
pub struct FadeOverlay;

/// Entity marker for the narration text node.
#[derive(Component)]
pub struct NarrationText;

/// Voice narration state — tracks which lines have been spoken.
#[derive(Resource, Default)]
pub struct VoiceNarration {
    /// Index of the last narration line that was triggered for audio playback.
    pub last_spoken: Option<usize>,
}

// ═══════════════════════════════════════════════════════════════════
// Setup — spawn the fade overlay and narration text
// ═══════════════════════════════════════════════════════════════════

pub fn setup_cinematic_overlay(mut commands: Commands) {
    // Fullscreen black fade overlay
    commands.spawn((
        Node {
            width: Val::Vw(100.0),
            height: Val::Vh(100.0),
            position_type: PositionType::Absolute,
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 1.0)),
        ZIndex(i32::MAX),
        FadeOverlay,
    ));

    // Narration text at bottom center
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(22.0),
            ..default()
        },
        TextColor(Color::srgba(0.85, 0.9, 0.95, 0.0)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(60.0),
            left: Val::Percent(10.0),
            right: Val::Percent(10.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        ZIndex(i32::MAX - 1),
        NarrationText,
    ));
}

// ═══════════════════════════════════════════════════════════════════
// Screen fade system
// ═══════════════════════════════════════════════════════════════════

pub fn screen_fade_system(
    time: Res<Time>,
    mut fade: ResMut<ScreenFade>,
    mut overlays: Query<&mut BackgroundColor, With<FadeOverlay>>,
) {
    let dt = time.delta_secs();
    let diff = fade.target - fade.alpha;
    if diff.abs() > 0.001 {
        fade.alpha += diff.signum() * fade.speed * dt;
        fade.alpha = fade.alpha.clamp(0.0, 1.0);
    } else {
        fade.alpha = fade.target;
    }

    for mut bg in overlays.iter_mut() {
        *bg = BackgroundColor(Color::srgba(0.0, 0.0, 0.0, fade.alpha));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Narration system
// ═══════════════════════════════════════════════════════════════════

pub fn narration_system(
    time: Res<Time>,
    director: Res<CinematicDirector>,
    mut narration: ResMut<NarrationState>,
    mut texts: Query<(&mut Text, &mut TextColor), With<NarrationText>>,
) {
    if !director.enabled {
        return;
    }

    let t = director.total_time;
    let mut active_text = "";
    let mut text_progress = 0.0f32;

    for &(start, end, text) in NARRATION {
        if t >= start && t <= end {
            active_text = text;
            let duration = end - start;
            let elapsed = t - start;
            // Fade in first 0.5s, hold, fade out last 0.5s
            text_progress = if elapsed < 0.5 {
                elapsed / 0.5
            } else if elapsed > duration - 0.5 {
                (duration - elapsed) / 0.5
            } else {
                1.0
            };
            break;
        }
    }

    narration.current_text = active_text.to_string();
    narration.text_alpha = text_progress;

    for (mut text, mut color) in texts.iter_mut() {
        if active_text.is_empty() {
            *text = Text::new("");
        } else {
            *text = Text::new(active_text);
        }
        *color = TextColor(Color::srgba(0.85, 0.9, 0.95, text_progress));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cinematic director — the main orchestrator
// ═══════════════════════════════════════════════════════════════════

/// Voice narration trigger — logs when each narration line starts.
/// When audio is enabled (live-audio feature), this will trigger playback.
/// Pre-render narration WAVs with: symthaea-voice --formant --text "..."
pub fn voice_narration_system(director: Res<CinematicDirector>, mut voice: ResMut<VoiceNarration>) {
    if !director.enabled {
        return;
    }

    let t = director.total_time;

    for (idx, &(start, _end, text)) in NARRATION.iter().enumerate() {
        if t >= start && t < start + 0.2 {
            if voice.last_spoken != Some(idx) {
                voice.last_spoken = Some(idx);
                info!("[voice] t={:.1}s: \"{}\"", t, text);
                // TODO: when live-audio enabled, play pre-rendered audio:
                // let audio = asset_server.load(format!("narration/narration_{:02}.ogg", idx));
                // commands.spawn(AudioPlayer(audio));
            }
        }
    }
}

/// Cubic ease-in-out.
fn ease(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

pub fn cinematic_director_system(
    mut commands: Commands,
    time: Res<Time>,
    mut director: ResMut<CinematicDirector>,
    mut fade: ResMut<ScreenFade>,
    mut next_state: ResMut<NextState<GamePhase>>,
    current_state: Res<State<GamePhase>>,
    // Dungeon resources (optional — not present during globe)
    leviathan: Option<ResMut<crate::resources::LeviathanState>>,
    dim_transition: Option<ResMut<crate::systems::dimension_transition::DimensionTransition>>,
    // Globe resources (optional — not present during dungeon)
    camera_config: Option<ResMut<sol_atlas_bevy::camera::OrbitalCameraConfig>>,
    timeline: Option<ResMut<sol_atlas_bevy::timeline::TimelineState>>,
    aesthetic: Option<ResMut<crate::systems::atlas::CurrentAesthetic>>,
    // AI player
    ai_player: Option<ResMut<crate::systems::ai_player::AiPlayer>>,
) {
    if !director.enabled || director.phase == CinematicPhase::Done {
        return;
    }

    let dt = time.delta_secs();
    director.phase_timer += dt;
    director.total_time += dt;

    let t = (director.phase_timer / director.phase.duration()).clamp(0.0, 1.0);
    let e = ease(t);

    match director.phase {
        // ═══ ACT 1: THE ROOM ═══════════════════════════════════
        CinematicPhase::FadeIn => {
            // Fade from black — fast so dungeon is visible quickly
            fade.target = 0.0;
            fade.speed = 2.0;
            // Enable AI player
            if let Some(mut ai) = ai_player {
                ai.enabled = true;
            }
        }

        CinematicPhase::Explore => {
            // AI player explores naturally — consciousness auras + harmonies run automatically
            // Brighten the dungeon background slightly for cinematic visibility
            commands.insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.08)));
        }

        CinematicPhase::Leviathan => {
            // Escalate Leviathan to Stirring → Awake
            if let Some(mut lev) = leviathan {
                if t < 0.5 {
                    lev.phase = crate::resources::SleepPhase::Stirring;
                } else {
                    lev.phase = crate::resources::SleepPhase::Awake;
                }
            }
        }

        CinematicPhase::Dimension => {
            // Transition to 4D
            if let Some(mut dim) = dim_transition {
                dim.target = crate::systems::dimension_transition::DimensionMode::D4;
                // Sweep W position
                dim.w_position = (t * std::f32::consts::TAU).sin() * 50.0;
            }
            // Calm Leviathan for the dimension exploration
            if let Some(mut lev) = leviathan {
                lev.phase = crate::resources::SleepPhase::Stirring;
            }
        }

        CinematicPhase::Calm => {
            // Return to 2D, calm Leviathan
            if let Some(mut dim) = dim_transition {
                dim.target = crate::systems::dimension_transition::DimensionMode::D2;
                dim.w_position = 0.0;
            }
            if let Some(mut lev) = leviathan {
                lev.phase = crate::resources::SleepPhase::Dormant;
            }
        }

        // ═══ ACT 2: THE PLANET ═════════════════════════════════
        CinematicPhase::FadeToGlobe => {
            if !director.fade_triggered {
                fade.target = 1.0;
                fade.speed = 1.5;
                director.fade_triggered = true;
            }
            // When fully black, switch to globe
            if fade.alpha > 0.95 && *current_state.get() == GamePhase::Playing {
                next_state.set(GamePhase::GlobeView);
                // Disable AI player during globe
                if let Some(mut ai) = ai_player {
                    ai.enabled = false;
                }
            }
            // Start fading back in once in globe
            if *current_state.get() == GamePhase::GlobeView && fade.alpha > 0.9 {
                fade.target = 0.0;
                fade.speed = 1.0;
            }
        }

        CinematicPhase::GlobeHolo => {
            director.fade_triggered = false;
            if let Some(mut cam) = camera_config {
                cam.auto_rotate = true;
                cam.distance = 4.5;
            }
        }

        CinematicPhase::SplitScreen => {
            // TODO: spawn PiP dungeon camera viewport
            // For now, just continue globe rotation
            if let Some(mut cam) = camera_config {
                cam.theta += 0.001;
            }
        }

        CinematicPhase::Timeline => {
            if let Some(mut tl) = timeline {
                tl.year = (e * 200.0) as u32;
            }
            if let Some(mut cam) = camera_config {
                cam.distance = 3.5;
                cam.theta += 0.0003;
            }
        }

        CinematicPhase::Satellites => {
            // Orbital tracks drawn by orbital_tracks_system (gizmos)
            if let Some(mut cam) = camera_config {
                cam.distance = 4.0;
                cam.theta += 0.0008;
            }
        }

        CinematicPhase::Night => {
            if let Some(mut aes) = aesthetic {
                if aes.aesthetic != sol_atlas_core::aesthetics::Aesthetic::Night {
                    aes.aesthetic = sol_atlas_core::aesthetics::Aesthetic::Night;
                    aes.changed = true;
                }
            }
            if let Some(mut cam) = camera_config {
                cam.theta += 0.0005;
            }
        }

        // ═══ ACT 3: THE SOLAR SYSTEM ═══════════════════════════
        CinematicPhase::PullOut => {
            // Pull camera far back — Earth shrinks, planets visible
            if let Some(mut cam) = camera_config {
                cam.auto_rotate = false;
                cam.distance = 4.5 + e * 12.0; // 4.5 → 16.5
                cam.theta += 0.0005;
            }
            // Return to holographic
            if let Some(mut aes) = aesthetic {
                if aes.aesthetic != sol_atlas_core::aesthetics::Aesthetic::Holographic {
                    aes.aesthetic = sol_atlas_core::aesthetics::Aesthetic::Holographic;
                    aes.changed = true;
                }
            }
        }

        CinematicPhase::FlyToMars => {
            // Fly toward Mars position
            if let Some(mut cam) = camera_config {
                if let Some(mars) = sol_atlas_core::solar_system::solar_system_bodies()
                    .into_iter()
                    .find(|b| b.name == "Mars")
                {
                    let pos =
                        sol_atlas_core::solar_system::body_position(&mars, director.total_time);
                    // Interpolate camera toward Mars
                    let target_theta = pos[2].atan2(pos[0]);
                    let target_dist = (pos[0] * pos[0] + pos[2] * pos[2]).sqrt() - 0.5;
                    cam.theta += (target_theta - cam.theta) * 0.03;
                    cam.distance += (target_dist - cam.distance) * 0.03;
                    cam.phi += (pos[1].atan2(target_dist) - cam.phi) * 0.02;
                }
            }
        }

        CinematicPhase::FlyToJupiter => {
            if let Some(mut cam) = camera_config {
                if let Some(jupiter) = sol_atlas_core::solar_system::solar_system_bodies()
                    .into_iter()
                    .find(|b| b.name == "Jupiter")
                {
                    let pos =
                        sol_atlas_core::solar_system::body_position(&jupiter, director.total_time);
                    let target_theta = pos[2].atan2(pos[0]);
                    let target_dist = (pos[0] * pos[0] + pos[2] * pos[2]).sqrt() - 1.0;
                    cam.theta += (target_theta - cam.theta) * 0.03;
                    cam.distance += (target_dist - cam.distance) * 0.03;
                    cam.phi += (0.0 - cam.phi) * 0.02;
                }
            }
        }

        CinematicPhase::SaturnRings => {
            if let Some(mut cam) = camera_config {
                if let Some(saturn) = sol_atlas_core::solar_system::solar_system_bodies()
                    .into_iter()
                    .find(|b| b.name == "Saturn")
                {
                    let pos =
                        sol_atlas_core::solar_system::body_position(&saturn, director.total_time);
                    let target_theta = pos[2].atan2(pos[0]);
                    let target_dist = (pos[0] * pos[0] + pos[2] * pos[2]).sqrt() - 1.0;
                    cam.theta += (target_theta - cam.theta) * 0.03;
                    cam.distance += (target_dist - cam.distance) * 0.03;
                }
            }
        }

        CinematicPhase::ReturnEarth => {
            // Fly back to Earth
            if let Some(mut cam) = camera_config {
                cam.theta += (0.0 - cam.theta) * 0.05;
                cam.phi += (0.1 - cam.phi) * 0.03;
                cam.distance += (4.5 - cam.distance) * 0.04;
            }
        }

        // ═══ ACT 4: CLOSING ════════════════════════════════════
        CinematicPhase::FadeToRoom => {
            if !director.fade_triggered {
                fade.target = 1.0;
                fade.speed = 1.5;
                director.fade_triggered = true;
            }
            if fade.alpha > 0.95 && *current_state.get() == GamePhase::GlobeView {
                next_state.set(GamePhase::Playing);
                if let Some(mut ai) = ai_player {
                    ai.enabled = true;
                }
            }
            if *current_state.get() == GamePhase::Playing && fade.alpha > 0.9 {
                fade.target = 0.0;
                fade.speed = 1.0;
            }
        }

        CinematicPhase::Dungeon3D => {
            director.fade_triggered = false;
            if let Some(mut dim) = dim_transition {
                dim.target = crate::systems::dimension_transition::DimensionMode::D3;
            }
            // Brief Leviathan awake for dramatic CRT
            if let Some(mut lev) = leviathan {
                if t > 0.3 && t < 0.7 {
                    lev.phase = crate::resources::SleepPhase::Awake;
                } else {
                    lev.phase = crate::resources::SleepPhase::Dormant;
                }
            }
        }

        CinematicPhase::FinalFade => {
            // Return to 2D, slow fade to black
            if let Some(mut dim) = dim_transition {
                dim.target = crate::systems::dimension_transition::DimensionMode::D2;
            }
            fade.target = 1.0;
            fade.speed = 0.3; // slow dramatic fade
        }

        CinematicPhase::Done => {}
    }

    // Phase transition
    if director.phase_timer >= director.phase.duration() {
        let old = director.phase;
        director.phase = director.phase.next();
        director.phase_timer = 0.0;
        if director.phase != CinematicPhase::Done {
            info!(
                "[cinematic] {:?} → {:?} (t={:.1}s)",
                old, director.phase, director.total_time
            );
        } else {
            info!(
                "[cinematic] Complete! {:.1}s total. Stitch: ffmpeg -framerate 8 -i /tmp/symtropy-cinematic-frames/frame_%05d.png -c:v libx264 cinematic.mp4",
                director.total_time
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Orbital track gizmos — satellite shells around Earth
// ═══════════════════════════════════════════════════════════════════

pub fn orbital_tracks_system(
    mut gizmos: Gizmos,
    time: Res<Time>,
    director: Res<CinematicDirector>,
) {
    if !director.enabled {
        return;
    }
    // Only draw during Satellites phase (and Night for persistence)
    if !matches!(
        director.phase,
        CinematicPhase::Satellites | CinematicPhase::Night
    ) {
        return;
    }

    let t = time.elapsed_secs();

    // LEO shell (400km ≈ 1.06 Earth radii)
    let leo_r = 1.06f32;
    let leo_color = Color::linear_rgba(0.0, 0.8, 1.0, 0.15);
    // MEO shell (20,000km ≈ 4.1 Earth radii)
    let meo_r = 1.25f32; // compressed for visual clarity
    let meo_color = Color::linear_rgba(0.5, 1.0, 0.3, 0.1);
    // GEO shell (36,000km ≈ 6.6 Earth radii)
    let geo_r = 1.6f32; // compressed
    let geo_color = Color::linear_rgba(1.0, 0.7, 0.2, 0.08);

    let shells = [
        (leo_r, leo_color, 48, 8),
        (meo_r, meo_color, 36, 4),
        (geo_r, geo_color, 24, 2),
    ];

    for (radius, color, segments, sat_count) in shells {
        // Draw orbital ring
        for i in 0..segments {
            let a0 = i as f32 / segments as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
            let p0 = Vec3::new(radius * a0.cos(), 0.0, radius * a0.sin());
            let p1 = Vec3::new(radius * a1.cos(), 0.0, radius * a1.sin());
            gizmos.line(p0, p1, color);
        }

        // Animated satellite dots
        for s in 0..sat_count {
            let speed = 0.3 / radius; // faster at lower orbits
            let angle = t * speed + s as f32 * std::f32::consts::TAU / sat_count as f32;
            let pos = Vec3::new(radius * angle.cos(), 0.02, radius * angle.sin());
            let bright_color = Color::linear_rgba(
                color.to_linear().red * 3.0,
                color.to_linear().green * 3.0,
                color.to_linear().blue * 3.0,
                0.8,
            );
            gizmos.sphere(Isometry3d::from_translation(pos), 0.008, bright_color);
        }
    }
}
