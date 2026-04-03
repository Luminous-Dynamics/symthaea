// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ghost echo system: "The room remembers you."
//!
//! Records player positions during gameplay and replays them as translucent
//! ghost sprites on seed replay. Creates the feeling of a living space
//! that has witnessed your previous attempts.

use bevy::prelude::*;

use crate::components::Player;
use crate::resources::GhostRecording;

/// Sample interval for ghost recording (seconds).
const GHOST_SAMPLE_INTERVAL: f32 = 1.0;

/// Marker for ghost echo sprites.
#[derive(Component)]
pub struct GhostSprite {
    pub frame_index: usize,
}

/// Resource holding loaded ghost data for replay.
#[derive(Resource, Default)]
pub struct GhostReplayData {
    pub frames: Vec<(f32, [f32; 2])>,
    pub current_index: usize,
    pub timer: f32,
    pub active: bool,
}

/// Record player position every ~1 second during gameplay.
pub fn ghost_record_system(
    mut recording: ResMut<GhostRecording>,
    player: Query<&Transform, With<Player>>,
    time: Res<Time>,
) {
    recording.sample_timer += time.delta_secs();
    if recording.sample_timer < GHOST_SAMPLE_INTERVAL {
        return;
    }
    recording.sample_timer = 0.0;

    if let Ok(tf) = player.single() {
        recording.frames.push((
            time.elapsed_secs(),
            [tf.translation.x, tf.translation.y],
        ));
    }
}

/// Save ghost recording to disk when the game ends.
pub fn ghost_save_system(recording: &GhostRecording, seed: u64) {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let dir = std::path::PathBuf::from(home).join(".symtropy");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("ghost_{seed}.json"));

    match serde_json::to_string(&recording.frames) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                eprintln!("[ghost] Failed to save: {e}");
            } else {
                eprintln!("[ghost] Saved {} frames to {}", recording.frames.len(), path.display());
            }
        }
        Err(e) => eprintln!("[ghost] Failed to serialize: {e}"),
    }
}

/// Load ghost recording from disk for a given seed.
#[allow(dead_code)] // Called from plugin.rs game_over/victory when replaying
pub fn load_ghost_for_seed(seed: u64) -> Option<Vec<(f32, [f32; 2])>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let path = std::path::PathBuf::from(home)
        .join(".symtropy")
        .join(format!("ghost_{seed}.json"));

    let json = std::fs::read_to_string(&path).ok()?;
    let frames: Vec<(f32, [f32; 2])> = serde_json::from_str(&json).ok()?;
    if frames.is_empty() {
        return None;
    }
    eprintln!("[ghost] Loaded {} frames for seed {seed}", frames.len());
    Some(frames)
}

/// Spawn ghost sprites from loaded replay data.
pub fn ghost_spawn_system(
    mut commands: Commands,
    replay: Res<GhostReplayData>,
    existing: Query<Entity, With<GhostSprite>>,
) {
    // Only spawn once when replay data first becomes active.
    if !replay.active || !existing.is_empty() {
        return;
    }

    // Spawn a single ghost sprite that will be moved through the recorded positions.
    commands.spawn((
        Sprite::from_color(
            Color::srgba(0.5, 0.8, 0.9, 0.15), // translucent cyan ghost
            Vec2::splat(16.0),
        ),
        Transform::from_xyz(0.0, 0.0, 1.5), // between floor and entities
        GhostSprite { frame_index: 0 },
    ));
}

/// Animate ghost sprite through recorded positions.
pub fn ghost_replay_system(
    mut replay: ResMut<GhostReplayData>,
    mut ghosts: Query<(&mut Transform, &mut GhostSprite, &mut Sprite)>,
    time: Res<Time>,
) {
    if !replay.active || replay.frames.is_empty() {
        return;
    }

    replay.timer += time.delta_secs();

    // Advance through frames at recording rate.
    while replay.current_index < replay.frames.len() {
        let (frame_time, _) = replay.frames[replay.current_index];
        let first_time = replay.frames[0].0;
        if replay.timer < frame_time - first_time {
            break;
        }
        replay.current_index += 1;
    }

    // Position ghost at current frame.
    let idx = replay.current_index.min(replay.frames.len() - 1);
    let (_, pos) = replay.frames[idx];

    for (mut tf, mut ghost, mut sprite) in &mut ghosts {
        tf.translation.x = pos[0];
        tf.translation.y = pos[1];
        ghost.frame_index = idx;

        // Fade out as replay nears end.
        let progress = idx as f32 / replay.frames.len() as f32;
        let alpha = 0.15 * (1.0 - progress * 0.5); // fade from 0.15 to 0.075
        sprite.color = Color::srgba(0.5, 0.8, 0.9, alpha);
    }

    // Deactivate when replay is complete.
    if replay.current_index >= replay.frames.len() {
        replay.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_data_serialization_roundtrip() {
        let frames: Vec<(f32, [f32; 2])> = vec![
            (0.0, [10.0, 20.0]),
            (1.0, [15.0, 25.0]),
            (2.0, [20.0, 30.0]),
        ];

        let json = serde_json::to_string(&frames).unwrap();
        let loaded: Vec<(f32, [f32; 2])> = serde_json::from_str(&json).unwrap();

        assert_eq!(frames.len(), loaded.len());
        for (a, b) in frames.iter().zip(loaded.iter()) {
            assert!((a.0 - b.0).abs() < 1e-6);
            assert!((a.1[0] - b.1[0]).abs() < 1e-6);
            assert!((a.1[1] - b.1[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_frames_no_load() {
        let result = load_ghost_for_seed(999999999); // nonexistent seed
        assert!(result.is_none());
    }

    #[test]
    fn replay_data_defaults_inactive() {
        let replay = GhostReplayData::default();
        assert!(!replay.active);
        assert!(replay.frames.is_empty());
    }
}
