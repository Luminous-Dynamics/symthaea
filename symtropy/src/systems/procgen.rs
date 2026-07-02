// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Procedural level generation using BSP (Binary Space Partition).
//!
//! Generates a random dungeon layout each run:
//! 1. Recursively split the map into rooms (BSP tree)
//! 2. Carve rooms at leaf nodes
//! 3. Connect siblings with corridors
//! 4. Place player start and fusion core at maximum distance

use rand::Rng;

/// Generated dungeon layout.
pub struct Dungeon {
    pub width: usize,
    pub height: usize,
    /// 0=wall, 1=floor, 2=core_room, 3=player_start
    pub tiles: Vec<Vec<u8>>,
    /// Room centers for energy well placement.
    pub room_centers: Vec<(usize, usize)>,
}

struct BspNode {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    left: Option<Box<BspNode>>,
    right: Option<Box<BspNode>>,
    room: Option<Room>,
}

struct Room {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Room {
    fn center(&self) -> (usize, usize) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }
}

const DEFAULT_MIN_ROOM_SIZE: usize = 5;
const DEFAULT_MIN_SPLIT_SIZE: usize = 12;
const DEFAULT_BSP_DEPTH: usize = 5;

impl BspNode {
    fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self {
            x,
            y,
            w,
            h,
            left: None,
            right: None,
            room: None,
        }
    }

    fn split_with_params(
        &mut self,
        rng: &mut impl Rng,
        depth: usize,
        min_room: usize,
        min_split: usize,
    ) {
        if depth == 0 || (self.w < min_split && self.h < min_split) {
            // Leaf — carve a room
            let room_w = rng.gen_range(min_room..=self.w.saturating_sub(2).max(min_room));
            let room_h = rng.gen_range(min_room..=self.h.saturating_sub(2).max(min_room));
            let room_x = self.x + rng.gen_range(1..=self.w.saturating_sub(room_w).max(1));
            let room_y = self.y + rng.gen_range(1..=self.h.saturating_sub(room_h).max(1));
            self.room = Some(Room {
                x: room_x,
                y: room_y,
                w: room_w,
                h: room_h,
            });
            return;
        }

        let split_horizontal = if self.w > self.h * 2 {
            false
        } else if self.h > self.w * 2 {
            true
        } else {
            rng.gen_bool(0.5)
        };

        if split_horizontal {
            let split = rng.gen_range(min_room..=self.h.saturating_sub(min_room).max(min_room));
            let mut left = Box::new(BspNode::new(self.x, self.y, self.w, split));
            let mut right = Box::new(BspNode::new(
                self.x,
                self.y + split,
                self.w,
                self.h.saturating_sub(split).max(1),
            ));
            left.split_with_params(rng, depth - 1, min_room, min_split);
            right.split_with_params(rng, depth - 1, min_room, min_split);
            self.left = Some(left);
            self.right = Some(right);
        } else {
            let split = rng.gen_range(min_room..=self.w.saturating_sub(min_room).max(min_room));
            let mut left = Box::new(BspNode::new(self.x, self.y, split, self.h));
            let mut right = Box::new(BspNode::new(
                self.x + split,
                self.y,
                self.w.saturating_sub(split).max(1),
                self.h,
            ));
            left.split_with_params(rng, depth - 1, min_room, min_split);
            right.split_with_params(rng, depth - 1, min_room, min_split);
            self.left = Some(left);
            self.right = Some(right);
        }
    }

    fn carve(&self, tiles: &mut Vec<Vec<u8>>) {
        if let Some(ref room) = self.room {
            for y in room.y..room.y + room.h {
                for x in room.x..room.x + room.w {
                    if y < tiles.len() && x < tiles[0].len() {
                        tiles[y][x] = 1; // floor
                    }
                }
            }
        }
        if let (Some(left), Some(right)) = (&self.left, &self.right) {
            left.carve(tiles);
            right.carve(tiles);
            // Connect siblings with a corridor
            if let (Some(lc), Some(rc)) = (left.find_room_center(), right.find_room_center()) {
                carve_corridor(tiles, lc, rc);
            }
        }
    }

    fn find_room_center(&self) -> Option<(usize, usize)> {
        if let Some(ref room) = self.room {
            return Some(room.center());
        }
        // Try left child first, then right
        if let Some(ref left) = self.left {
            if let Some(c) = left.find_room_center() {
                return Some(c);
            }
        }
        if let Some(ref right) = self.right {
            return right.find_room_center();
        }
        None
    }

    fn collect_rooms(&self, rooms: &mut Vec<(usize, usize)>) {
        if let Some(ref room) = self.room {
            rooms.push(room.center());
        }
        if let Some(ref left) = self.left {
            left.collect_rooms(rooms);
        }
        if let Some(ref right) = self.right {
            right.collect_rooms(rooms);
        }
    }
}

fn carve_corridor(tiles: &mut Vec<Vec<u8>>, from: (usize, usize), to: (usize, usize)) {
    carve_corridor_width(tiles, from, to, 2);
}

fn carve_corridor_width(
    tiles: &mut Vec<Vec<u8>>,
    from: (usize, usize),
    to: (usize, usize),
    width: usize,
) {
    let (mut x, mut y) = from;
    let (tx, ty) = to;

    while x != tx {
        if x < tiles[0].len() && y < tiles.len() {
            tiles[y][x] = 1;
            for w in 1..width {
                if y + w < tiles.len() {
                    tiles[y + w][x] = 1;
                }
            }
        }
        if x < tx {
            x += 1;
        } else {
            x -= 1;
        }
    }
    while y != ty {
        if x < tiles[0].len() && y < tiles.len() {
            tiles[y][x] = 1;
            for w in 1..width {
                if x + w < tiles[0].len() {
                    tiles[y][x + w] = 1;
                }
            }
        }
        if y < ty {
            y += 1;
        } else {
            y -= 1;
        }
    }
}

/// Generate a random dungeon.
pub fn generate_dungeon(width: usize, height: usize, seed: u64) -> Dungeon {
    generate_dungeon_with_config(
        width,
        height,
        seed,
        DEFAULT_BSP_DEPTH,
        DEFAULT_MIN_ROOM_SIZE,
        DEFAULT_MIN_SPLIT_SIZE,
        2,
    )
}

/// Generate a consciousness-modulated dungeon.
///
/// Uses PhiDungeonConfig to vary BSP depth, room sizes, corridor width,
/// and extra connections based on the player's consciousness level.
pub fn generate_dungeon_phi(
    width: usize,
    height: usize,
    seed: u64,
    config: &super::phi_pcg::PhiDungeonConfig,
) -> Dungeon {
    let mut dungeon = generate_dungeon_with_config(
        width,
        height,
        seed,
        config.bsp_depth,
        config.min_room_size,
        config.min_split_size,
        config.corridor_width,
    );

    // Add extra corridor connections for high-Phi (integration loops)
    if config.extra_connections > 0 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(1000));
        let mut root = BspNode::new(0, 0, width, height);
        root.split_with_params(
            &mut rng,
            config.bsp_depth,
            config.min_room_size,
            config.min_split_size,
        );
        let mut rooms = Vec::new();
        root.collect_rooms(&mut rooms);

        // Connect non-adjacent rooms to create loops (integration)
        for i in 0..config.extra_connections.min(rooms.len().saturating_sub(2)) {
            let from_idx = i;
            let to_idx = (i + 2).min(rooms.len() - 1);
            if from_idx < rooms.len() && to_idx < rooms.len() {
                carve_corridor_width(
                    &mut dungeon.tiles,
                    rooms[from_idx],
                    rooms[to_idx],
                    config.corridor_width,
                );
            }
        }
    }

    dungeon
}

fn generate_dungeon_with_config(
    width: usize,
    height: usize,
    seed: u64,
    bsp_depth: usize,
    min_room: usize,
    min_split: usize,
    _corridor_width: usize,
) -> Dungeon {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut tiles = vec![vec![0u8; width]; height]; // all walls

    // BSP split
    let mut root = BspNode::new(0, 0, width, height);
    root.split_with_params(&mut rng, bsp_depth, min_room, min_split);
    root.carve(&mut tiles);

    // Ensure border is all walls
    for x in 0..width {
        tiles[0][x] = 0;
        tiles[height - 1][x] = 0;
    }
    for y in 0..height {
        tiles[y][0] = 0;
        tiles[y][width - 1] = 0;
    }

    // Find rooms for player and core placement
    let mut rooms = Vec::new();
    root.collect_rooms(&mut rooms);

    if rooms.len() >= 2 {
        // Player at first room, core at last (maximum BSP distance)
        let player = rooms[0];
        let core = rooms[rooms.len() - 1];

        let py = player.1.clamp(1, height - 2);
        let px = player.0.clamp(1, width - 2);
        let cy = core.1.clamp(1, height - 2);
        let cx = core.0.clamp(1, width - 2);

        tiles[py][px] = 3; // player start
        tiles[cy][cx] = 2; // core room
        // Mark core room area
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let cy_near = (cy as i32 + dy) as usize;
                let cx_near = (cx as i32 + dx) as usize;
                if cy_near < height && cx_near < width && tiles[cy_near][cx_near] == 1 {
                    tiles[cy_near][cx_near] = 2;
                }
            }
        }
    } else if !rooms.is_empty() {
        let py = rooms[0].1.clamp(1, height - 2);
        let px = rooms[0].0.clamp(1, width - 2);
        tiles[py][px] = 3;
        // Place core somewhere walkable
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if tiles[y][x] == 1 {
                    tiles[y][x] = 2;
                    break;
                }
            }
        }
    }

    Dungeon {
        width,
        height,
        tiles,
        room_centers: rooms,
    }
}

/// Generate and insert SiteLayout and TileGrid resources.
pub fn generate_site_layout_system(
    mut commands: bevy::prelude::Commands,
    seed: bevy::prelude::Res<crate::resources::DungeonSeed>,
    player_c: bevy::prelude::Res<crate::systems::consciousness::PlayerConsciousness>,
) {
    use crate::resources::{SiteLayout, TileGrid};
    use crate::systems::rendering::{MAP_HEIGHT, MAP_WIDTH, TILE_SIZE};
    use bevy::prelude::*;

    let phi = player_c.level;

    // level_dungeon inline:
    let dungeon = {
        let config = crate::systems::phi_pcg::PhiDungeonConfig::from_phi(
            &crate::systems::phi_pcg::PhiPcgParams {
                phi,
                ..Default::default()
            },
        );
        info!(
            "[symtropy] Phi-PCG: phi={:.2}, depth={}, rooms_min={}, connections={}",
            phi, config.bsp_depth, config.min_room_size, config.extra_connections
        );
        generate_dungeon_phi(MAP_WIDTH as usize, MAP_HEIGHT as usize, seed.0, &config)
    };

    info!(
        "[symtropy] Generated dungeon with seed {} ({} rooms)",
        seed.0,
        dungeon.room_centers.len()
    );

    let map = &dungeon.tiles;
    let rows = map.len() as i32;
    let cols = if map.is_empty() {
        0
    } else {
        map[0].len() as i32
    };

    // Build tile grid for O(1) collision lookups
    let mut tile_grid = TileGrid {
        tile_size: TILE_SIZE,
        origin_col: cols / 2,
        origin_row: rows / 2,
        cols,
        rows,
        ..default()
    };

    let mut site_layout = SiteLayout {
        site_id: "seedworks_firstlight".to_string(),
        width: cols as usize,
        height: rows as usize,
        tiles: dungeon.tiles.clone(),
        room_centers: dungeon.room_centers.clone(),
        player_start: Vec2::ZERO,
        core_pos: Vec2::ZERO,
    };

    // Calculate positions and populate TileGrid
    for (row_idx, row) in map.iter().enumerate() {
        for (col_idx, &cell) in row.iter().enumerate() {
            let x = (col_idx as f32 - cols as f32 / 2.0) * TILE_SIZE;
            let y = (rows as f32 / 2.0 - row_idx as f32) * TILE_SIZE;

            let walkable = cell != 1;
            if cell == 2 {
                site_layout.core_pos = Vec2::new(x, y);
            } else if cell == 3 {
                site_layout.player_start = Vec2::new(x, y);
            }

            tile_grid
                .cells
                .insert((col_idx as i32, row_idx as i32), walkable);
        }
    }

    commands.insert_resource(tile_grid);
    commands.insert_resource(site_layout);
}

use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_valid_dungeon() {
        let dungeon = generate_dungeon(30, 22, 42);
        assert_eq!(dungeon.width, 30);
        assert_eq!(dungeon.height, 22);

        // Has at least some floor tiles
        let floor_count: usize = dungeon
            .tiles
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&t| t > 0)
            .count();
        assert!(
            floor_count > 20,
            "should have walkable tiles, got {}",
            floor_count
        );

        // Has player start
        let has_player = dungeon
            .tiles
            .iter()
            .flat_map(|row| row.iter())
            .any(|&t| t == 3);
        assert!(has_player, "should have player start");

        // Has core
        let has_core = dungeon
            .tiles
            .iter()
            .flat_map(|row| row.iter())
            .any(|&t| t == 2);
        assert!(has_core, "should have core room");

        // Border is walls
        for x in 0..30 {
            assert_eq!(dungeon.tiles[0][x], 0);
            assert_eq!(dungeon.tiles[21][x], 0);
        }
    }

    #[test]
    fn different_seeds_different_layouts() {
        let d1 = generate_dungeon(30, 22, 1);
        let d2 = generate_dungeon(30, 22, 2);
        let tiles1: Vec<u8> = d1.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        let tiles2: Vec<u8> = d2.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        assert_ne!(tiles1, tiles2);
    }

    #[test]
    fn phi_driven_generation_varies_with_phi() {
        use super::super::phi_pcg::{PhiDungeonConfig, PhiPcgParams};

        let low_phi_config = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.1,
            ..Default::default()
        });
        let high_phi_config = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.9,
            ..Default::default()
        });

        let low = generate_dungeon_phi(30, 22, 42, &low_phi_config);
        let high = generate_dungeon_phi(30, 22, 42, &high_phi_config);

        let low_tiles: Vec<u8> = low.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        let high_tiles: Vec<u8> = high.tiles.iter().flat_map(|r| r.iter().copied()).collect();

        // High Phi and low Phi should produce DIFFERENT layouts (different params)
        assert_ne!(
            low_tiles, high_tiles,
            "different phi levels should produce different layouts"
        );

        // Verify BSP depth difference is reflected in the config
        assert!(high_phi_config.bsp_depth > low_phi_config.bsp_depth);
        assert!(high_phi_config.extra_connections > low_phi_config.extra_connections);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let d1 = generate_dungeon(30, 22, 42);
        let d2 = generate_dungeon(30, 22, 42);
        let tiles1: Vec<u8> = d1.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        let tiles2: Vec<u8> = d2.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        assert_eq!(tiles1, tiles2);
    }
}
