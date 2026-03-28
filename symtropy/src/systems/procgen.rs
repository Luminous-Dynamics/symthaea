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

const MIN_ROOM_SIZE: usize = 5;
const MIN_SPLIT_SIZE: usize = 12;

impl BspNode {
    fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self { x, y, w, h, left: None, right: None, room: None }
    }

    fn split(&mut self, rng: &mut impl Rng, depth: usize) {
        if depth == 0 || (self.w < MIN_SPLIT_SIZE && self.h < MIN_SPLIT_SIZE) {
            // Leaf — carve a room
            let room_w = rng.gen_range(MIN_ROOM_SIZE..=self.w.saturating_sub(2).max(MIN_ROOM_SIZE));
            let room_h = rng.gen_range(MIN_ROOM_SIZE..=self.h.saturating_sub(2).max(MIN_ROOM_SIZE));
            let room_x = self.x + rng.gen_range(1..=self.w.saturating_sub(room_w).max(1));
            let room_y = self.y + rng.gen_range(1..=self.h.saturating_sub(room_h).max(1));
            self.room = Some(Room { x: room_x, y: room_y, w: room_w, h: room_h });
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
            let split = rng.gen_range(MIN_ROOM_SIZE..=self.h.saturating_sub(MIN_ROOM_SIZE).max(MIN_ROOM_SIZE));
            let mut left = Box::new(BspNode::new(self.x, self.y, self.w, split));
            let mut right = Box::new(BspNode::new(self.x, self.y + split, self.w, self.h.saturating_sub(split).max(1)));
            left.split(rng, depth - 1);
            right.split(rng, depth - 1);
            self.left = Some(left);
            self.right = Some(right);
        } else {
            let split = rng.gen_range(MIN_ROOM_SIZE..=self.w.saturating_sub(MIN_ROOM_SIZE).max(MIN_ROOM_SIZE));
            let mut left = Box::new(BspNode::new(self.x, self.y, split, self.h));
            let mut right = Box::new(BspNode::new(self.x + split, self.y, self.w.saturating_sub(split).max(1), self.h));
            left.split(rng, depth - 1);
            right.split(rng, depth - 1);
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
        if let (Some(ref left), Some(ref right)) = (&self.left, &self.right) {
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
        if let Some(ref left) = self.left { left.collect_rooms(rooms); }
        if let Some(ref right) = self.right { right.collect_rooms(rooms); }
    }
}

fn carve_corridor(tiles: &mut Vec<Vec<u8>>, from: (usize, usize), to: (usize, usize)) {
    let (mut x, mut y) = from;
    let (tx, ty) = to;

    // L-shaped corridor: horizontal then vertical
    while x != tx {
        if x < tiles[0].len() && y < tiles.len() {
            tiles[y][x] = 1;
            // Make corridor 2 wide for easier navigation
            if y + 1 < tiles.len() { tiles[y + 1][x] = 1; }
        }
        if x < tx { x += 1; } else { x -= 1; }
    }
    while y != ty {
        if x < tiles[0].len() && y < tiles.len() {
            tiles[y][x] = 1;
            if x + 1 < tiles[0].len() { tiles[y][x + 1] = 1; }
        }
        if y < ty { y += 1; } else { y -= 1; }
    }
}

/// Generate a random dungeon.
pub fn generate_dungeon(width: usize, height: usize, seed: u64) -> Dungeon {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut tiles = vec![vec![0u8; width]; height]; // all walls

    // BSP split
    let mut root = BspNode::new(0, 0, width, height);
    root.split(&mut rng, 5); // depth 5 → ~32 rooms
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
        tiles[player.1][player.0] = 3; // player start
        tiles[core.1][core.0] = 2; // core room
        // Mark core room area
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let cy = (core.1 as i32 + dy) as usize;
                let cx = (core.0 as i32 + dx) as usize;
                if cy < height && cx < width && tiles[cy][cx] == 1 {
                    tiles[cy][cx] = 2;
                }
            }
        }
    } else if !rooms.is_empty() {
        tiles[rooms[0].1][rooms[0].0] = 3;
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

    Dungeon { width, height, tiles }
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
        let floor_count: usize = dungeon.tiles.iter()
            .flat_map(|row| row.iter())
            .filter(|&&t| t > 0)
            .count();
        assert!(floor_count > 20, "should have walkable tiles, got {}", floor_count);

        // Has player start
        let has_player = dungeon.tiles.iter()
            .flat_map(|row| row.iter())
            .any(|&t| t == 3);
        assert!(has_player, "should have player start");

        // Has core
        let has_core = dungeon.tiles.iter()
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
    fn deterministic_with_same_seed() {
        let d1 = generate_dungeon(30, 22, 42);
        let d2 = generate_dungeon(30, 22, 42);
        let tiles1: Vec<u8> = d1.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        let tiles2: Vec<u8> = d2.tiles.iter().flat_map(|r| r.iter().copied()).collect();
        assert_eq!(tiles1, tiles2);
    }
}
