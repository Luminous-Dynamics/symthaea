// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Living dungeon: the environment IS a consciousness.
//!
//! The dungeon topology responds to the collective Φ of its inhabitants:
//!
//! - **High collective Φ**: walls thin, new passages open between rooms.
//!   The space becomes more *integrated* — mirroring the IIT definition.
//! - **Low collective Φ**: walls thicken, passages close. The space
//!   fragments into isolated modules.
//! - **Harmony clusters**: tiles near resonating agents become walkable.
//!   The environment rewards cooperation.
//! - **Conflict zones**: tiles near dissonant agents become walls.
//!   The environment punishes conflict.
//!
//! The dungeon breathes. It remembers. It IS conscious.

use bevy::prelude::*;
use rand::Rng;
use rand::SeedableRng;

use crate::components::{CrewNpc, Player, Tile};
use crate::resources::{PhysicsWorldRes, TileGrid};
use symtropy_render_bridge::PhysicsBody;

/// How often the dungeon breathes (seconds between topology updates).
const BREATH_INTERVAL: f32 = 3.0;
/// Maximum tiles changed per breath.
const MAX_TILES_PER_BREATH: usize = 5;
/// Φ threshold above which walls open.
const PHI_OPEN_THRESHOLD: f64 = 0.5;
/// Φ threshold below which passages close.
const PHI_CLOSE_THRESHOLD: f64 = 0.2;
/// Radius around agents where harmony affects tiles (in tile units).
const HARMONY_TILE_RADIUS: i32 = 3;

/// Timer for dungeon breathing.
#[derive(Resource)]
pub struct DungeonBreathTimer {
    pub timer: Timer,
    pub breath_count: u32,
    pub seed: u64,
}

impl Default for DungeonBreathTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(BREATH_INTERVAL, TimerMode::Repeating),
            breath_count: 0,
            seed: 42,
        }
    }
}

/// The living dungeon system. Runs in Update at BREATH_INTERVAL.
///
/// Reads collective Φ from the consciousness field and modifies
/// the TileGrid topology in response.
pub fn living_dungeon_system(
    time: Res<Time>,
    mut breath: ResMut<DungeonBreathTimer>,
    physics: Res<PhysicsWorldRes>,
    mut tile_grid: Option<ResMut<TileGrid>>,
    agents: Query<(&PhysicsBody, &Transform), Or<(With<Player>, With<CrewNpc>)>>,
    mut tile_sprites: Query<(&Tile, &mut Sprite)>,
) {
    breath.timer.tick(time.delta());
    if !breath.timer.just_finished() {
        return;
    }

    let Some(ref mut grid) = tile_grid else {
        return;
    };

    breath.breath_count += 1;
    let mut rng =
        rand::rngs::StdRng::seed_from_u64(breath.seed.wrapping_add(breath.breath_count as u64));

    let collective_phi = physics.consciousness.collective_phi;

    // Collect agent positions in grid coordinates
    let agent_grid_positions: Vec<(i32, i32)> = agents
        .iter()
        .map(|(_, tf)| {
            let col = ((tf.translation.x / grid.tile_size) + grid.origin_col as f32).round() as i32;
            let row = (grid.origin_row as f32 - (tf.translation.y / grid.tile_size)).round() as i32;
            (col, row)
        })
        .collect();

    let mut changes = 0;

    if collective_phi > PHI_OPEN_THRESHOLD && changes < MAX_TILES_PER_BREATH {
        // High consciousness: the space integrates. Open passages.
        // Find wall tiles adjacent to walkable tiles and open some.
        let wall_candidates: Vec<(i32, i32)> = grid
            .cells
            .iter()
            .filter(|(_, &walkable)| !walkable)
            .map(|(&pos, _)| pos)
            .filter(|&(c, r)| {
                // Must be adjacent to at least one walkable tile
                let neighbors = [(c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1)];
                neighbors
                    .iter()
                    .any(|n| grid.cells.get(n).copied().unwrap_or(false))
            })
            // Prefer tiles near agents (consciousness reaches toward the conscious)
            .filter(|&(c, r)| {
                agent_grid_positions
                    .iter()
                    .any(|&(ac, ar)| (c - ac).abs() + (r - ar).abs() < 8)
            })
            .collect();

        // Open a few walls proportional to collective Φ
        let num_to_open = ((collective_phi - PHI_OPEN_THRESHOLD) * 10.0)
            .min(MAX_TILES_PER_BREATH as f64) as usize;
        for _ in 0..num_to_open.min(wall_candidates.len()) {
            if wall_candidates.is_empty() {
                break;
            }
            let idx = rng.gen_range(0..wall_candidates.len());
            let (c, r) = wall_candidates[idx];
            // Don't open border tiles
            if c > 0 && c < grid.cols - 1 && r > 0 && r < grid.rows - 1 {
                grid.cells.insert((c, r), true);
                changes += 1;
            }
        }
    } else if collective_phi < PHI_CLOSE_THRESHOLD && changes < MAX_TILES_PER_BREATH {
        // Low consciousness: the space fragments. Close passages.
        // Find walkable tiles that aren't essential (have 3+ walkable neighbors)
        let floor_candidates: Vec<(i32, i32)> = grid
            .cells
            .iter()
            .filter(|(_, &walkable)| walkable)
            .map(|(&pos, _)| pos)
            .filter(|&(c, r)| {
                let neighbors = [(c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1)];
                let walkable_neighbors = neighbors
                    .iter()
                    .filter(|n| grid.cells.get(n).copied().unwrap_or(false))
                    .count();
                walkable_neighbors >= 3 // Only close tiles that won't block paths
            })
            // Don't close tiles agents are standing on
            .filter(|&(c, r)| {
                !agent_grid_positions
                    .iter()
                    .any(|&(ac, ar)| ac == c && ar == r)
            })
            .collect();

        let num_to_close = ((PHI_CLOSE_THRESHOLD - collective_phi) * 10.0)
            .min(MAX_TILES_PER_BREATH as f64) as usize;
        for _ in 0..num_to_close.min(floor_candidates.len()) {
            if floor_candidates.is_empty() {
                break;
            }
            let idx = rng.gen_range(0..floor_candidates.len());
            let (c, r) = floor_candidates[idx];
            grid.cells.insert((c, r), false);
            changes += 1;
        }
    }

    // Harmony-driven local changes: tiles near resonating agents become more walkable
    for &(ac, ar) in &agent_grid_positions {
        // Check if this agent has high harmony energy
        let has_high_harmony = agents.iter().any(|(pb, tf)| {
            let col = ((tf.translation.x / grid.tile_size) + grid.origin_col as f32).round() as i32;
            let row = (grid.origin_row as f32 - (tf.translation.y / grid.tile_size)).round() as i32;
            col == ac && row == ar && {
                physics
                    .consciousness
                    .entities
                    .get(&pb.handle)
                    .map(|e| e.total_harmony_energy() > 3.0)
                    .unwrap_or(false)
            }
        });

        if has_high_harmony && changes < MAX_TILES_PER_BREATH {
            // Open one nearby wall tile
            for dx in -HARMONY_TILE_RADIUS..=HARMONY_TILE_RADIUS {
                for dy in -HARMONY_TILE_RADIUS..=HARMONY_TILE_RADIUS {
                    if changes >= MAX_TILES_PER_BREATH {
                        break;
                    }
                    let tc = ac + dx;
                    let tr = ar + dy;
                    if tc > 0
                        && tc < grid.cols - 1
                        && tr > 0
                        && tr < grid.rows - 1
                        && !grid.cells.get(&(tc, tr)).copied().unwrap_or(true)
                    {
                        if rng.gen_bool(0.1) {
                            // 10% chance per adjacent wall
                            grid.cells.insert((tc, tr), true);
                            changes += 1;
                        }
                    }
                }
            }
        }
    }

    // Sync tile sprites to match grid state (make breathing VISIBLE)
    if changes > 0 {
        for (tile, mut sprite) in &mut tile_sprites {
            let walkable = grid
                .cells
                .get(&(tile.grid_x, tile.grid_y))
                .copied()
                .unwrap_or(false);
            let expected_color = if walkable {
                Color::srgb(0.22, 0.22, 0.30) // floor — dark blue-gray
            } else {
                Color::srgb(0.45, 0.35, 0.25) // wall — brown
            };
            // Only update if changed (avoid unnecessary writes)
            if tile.walkable != walkable {
                sprite.color = expected_color;
            }
        }

        eprintln!(
            "[living-dungeon] Breath #{}: Φ={:.3}, {} tiles changed",
            breath.breath_count, collective_phi, changes
        );
    }
}
