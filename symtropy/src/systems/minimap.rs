// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimap: tracks explored tiles and shows player position.
//! Uses a resource to track exploration; visual overlay is handled by the HUD text.

use bevy::prelude::*;

use crate::components::Player;
use crate::resources::TileGrid;

/// Explore radius (tiles revealed around player).
const EXPLORE_RADIUS: i32 = 6;

/// Tracks which tiles have been explored.
#[derive(Resource)]
pub struct ExploredTiles {
    pub explored: Vec<Vec<bool>>,
    pub width: usize,
    pub height: usize,
    pub explore_count: usize,
    pub total_walkable: usize,
}

impl Default for ExploredTiles {
    fn default() -> Self {
        Self {
            explored: Vec::new(),
            width: 0,
            height: 0,
            explore_count: 0,
            total_walkable: 0,
        }
    }
}

/// Initialize explored tiles from the tile grid.
pub fn setup_minimap(mut explored: ResMut<ExploredTiles>, tile_grid: Option<Res<TileGrid>>) {
    let Some(ref grid) = tile_grid else { return };
    let w = grid.cols as usize;
    let h = grid.rows as usize;
    explored.width = w;
    explored.height = h;
    explored.explored = vec![vec![false; w]; h];
    explored.explore_count = 0;

    // Count total walkable tiles
    explored.total_walkable = grid.cells.values().filter(|&&w| w).count();
    info!(
        "Minimap initialized: {}x{} grid, {} walkable tiles",
        w, h, explored.total_walkable
    );
}

/// Reveal tiles around the player each frame.
pub fn update_minimap(
    player: Query<&Transform, With<Player>>,
    tile_grid: Option<Res<TileGrid>>,
    mut explored: ResMut<ExploredTiles>,
) {
    let Some(ref grid) = tile_grid else { return };
    let Ok(player_tf) = player.single() else {
        return;
    };

    let px = player_tf.translation.x;
    let py = player_tf.translation.y;
    let player_col = ((px / grid.tile_size) + grid.origin_col as f32).round() as i32;
    let player_row = (grid.origin_row as f32 - (py / grid.tile_size)).round() as i32;

    // Reveal tiles in radius around player
    for dy in -EXPLORE_RADIUS..=EXPLORE_RADIUS {
        for dx in -EXPLORE_RADIUS..=EXPLORE_RADIUS {
            if dx * dx + dy * dy <= EXPLORE_RADIUS * EXPLORE_RADIUS {
                let r = (player_row + dy) as usize;
                let c = (player_col + dx) as usize;
                if r < explored.height && c < explored.width && !explored.explored[r][c] {
                    explored.explored[r][c] = true;
                    // Only count walkable tiles as explored
                    if grid
                        .cells
                        .get(&(c as i32, r as i32))
                        .copied()
                        .unwrap_or(false)
                    {
                        explored.explore_count += 1;
                    }
                }
            }
        }
    }
}

impl ExploredTiles {
    /// Percentage of walkable tiles explored.
    pub fn explore_percent(&self) -> f32 {
        if self.total_walkable == 0 {
            return 0.0;
        }
        (self.explore_count as f32 / self.total_walkable as f32 * 100.0).min(100.0)
    }
}
