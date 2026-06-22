// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Ant Colony V0: stigmergic foraging loop.
//!
//! The colony is the primary cognitive organism. Individual ants are lightweight
//! local inferants operating against shared fields.

use std::collections::{HashMap, VecDeque};

use symtropy_lifesim_core::{DiffusionParams, FieldGrid, FieldLayer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntRole {
    Scout,
    Forager,
}

#[derive(Debug, Clone)]
pub struct AntCohort {
    pub x: usize,
    pub y: usize,
    pub role: AntRole,
    pub carrying_food: bool,
    pub energy: f32,
    pub precision: f32,
    pub exploration_bias: f32,
}

#[derive(Debug, Clone)]
pub struct ColonyMind {
    pub food_reserve: u32,
    pub scout_ratio: f32,
    pub stress: f32,
    pub trail_entropy: f32,
    pub pheromone_coherence: f32,
}

impl Default for ColonyMind {
    fn default() -> Self {
        Self {
            food_reserve: 0,
            scout_ratio: 0.25,
            stress: 0.0,
            trail_entropy: 1.0,
            pheromone_coherence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColonyDebugMetrics {
    pub tick: u64,
    pub food_reserve: u32,
    pub trail_entropy: f32,
    pub pheromone_coherence: f32,
    pub stress: f32,
}

#[derive(Debug, Clone)]
pub struct ColonyWorld {
    pub fields: FieldGrid,
    pub mind: ColonyMind,
    pub nest: (usize, usize),
    food_sources: HashMap<(usize, usize), u32>,
    ants: Vec<AntCohort>,
    tick: u64,
}

impl ColonyWorld {
    pub fn new(width: usize, height: usize, nest: (usize, usize), ant_count: usize) -> Self {
        let mut fields = FieldGrid::new(width, height);
        fields.set(FieldLayer::HomePheromone, nest.0, nest.1, 500.0);
        let mut ants = Vec::with_capacity(ant_count);
        for i in 0..ant_count {
            ants.push(AntCohort {
                x: nest.0,
                y: nest.1,
                role: if i % 4 == 0 {
                    AntRole::Scout
                } else {
                    AntRole::Forager
                },
                carrying_food: false,
                energy: 1.0,
                precision: 1.0,
                exploration_bias: 0.25 + (i % 7) as f32 * 0.03,
            });
        }

        Self {
            fields,
            mind: ColonyMind::default(),
            nest,
            food_sources: HashMap::new(),
            ants,
            tick: 0,
        }
    }

    pub fn ants(&self) -> &[AntCohort] {
        &self.ants
    }

    pub const fn tick(&self) -> u64 {
        self.tick
    }

    pub fn debug_metrics(&self) -> ColonyDebugMetrics {
        ColonyDebugMetrics {
            tick: self.tick,
            food_reserve: self.mind.food_reserve,
            trail_entropy: self.mind.trail_entropy,
            pheromone_coherence: self.mind.pheromone_coherence,
            stress: self.mind.stress,
        }
    }

    pub fn add_food_source(&mut self, x: usize, y: usize, amount: u32) {
        self.food_sources.insert((x, y), amount);
    }

    pub fn set_obstacle(&mut self, x: usize, y: usize, blocked: bool) {
        self.fields
            .set(FieldLayer::Obstacle, x, y, if blocked { 1.0 } else { 0.0 });
    }

    pub fn add_danger(&mut self, x: usize, y: usize, intensity: f32) {
        self.fields
            .add(FieldLayer::DangerPheromone, x, y, intensity);
    }

    /// Couple the colony to a basin-scale substrate.
    ///
    /// Basin toxin and standing moisture become ant danger, so a colony can
    /// reroute without depending directly on `symtropy-basin` types.
    pub fn absorb_basin_fields(&mut self, basin_fields: &FieldGrid) {
        assert_eq!(
            self.fields.width(),
            basin_fields.width(),
            "field widths differ"
        );
        assert_eq!(
            self.fields.height(),
            basin_fields.height(),
            "field heights differ"
        );

        for y in 0..self.fields.height() {
            for x in 0..self.fields.width() {
                let toxin = basin_fields.get(FieldLayer::Toxin, x, y);
                let moisture = basin_fields.get(FieldLayer::Moisture, x, y);
                let flood_pressure = (moisture - 35.0).max(0.0) * 0.08;
                let danger = toxin * 0.06 + flood_pressure;
                if danger > 0.0 {
                    self.fields.add(FieldLayer::DangerPheromone, x, y, danger);
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.tick += 1;
        self.reinforce_sources();

        let params = DiffusionParams {
            diffusion: 0.08,
            decay: 0.035,
            dt: 1.0,
            max_value: 1_000.0,
        };
        self.fields
            .step_diffuse_decay(FieldLayer::FoodPheromone, params);
        self.fields
            .step_diffuse_decay(FieldLayer::HomePheromone, params);
        self.fields.step_diffuse_decay(
            FieldLayer::DangerPheromone,
            DiffusionParams {
                decay: 0.02,
                ..params
            },
        );

        let mut delivered = 0;
        let width = self.fields.width();
        let height = self.fields.height();
        for i in 0..self.ants.len() {
            let target = self.next_target(i, width, height);
            let ant = &mut self.ants[i];
            ant.x = target.0;
            ant.y = target.1;

            if ant.carrying_food {
                if (ant.x, ant.y) == self.nest {
                    ant.carrying_food = false;
                    delivered += 1;
                } else {
                    self.fields
                        .add(FieldLayer::FoodPheromone, ant.x, ant.y, 4.0);
                }
            } else {
                self.fields
                    .add(FieldLayer::HomePheromone, ant.x, ant.y, 1.5);
                if let Some(food) = self.food_sources.get_mut(&(ant.x, ant.y)) {
                    if *food > 0 {
                        *food -= 1;
                        ant.carrying_food = true;
                    }
                }
            }
        }

        self.mind.food_reserve += delivered;
        self.update_mind();
    }

    pub fn run_steps(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    pub fn traffic_in_danger_region(&self, min_x: usize, max_x: usize) -> usize {
        self.ants
            .iter()
            .filter(|ant| ant.x >= min_x && ant.x <= max_x)
            .count()
    }

    /// Export a simple PPM heatmap for debug visualization.
    ///
    /// This is intentionally dependency-free. It gives tests, scripts, and early
    /// design tools a concrete way to inspect field organization before a Bevy
    /// or WGSL viewer exists.
    pub fn to_ppm_heatmap(&self) -> String {
        let width = self.fields.width();
        let height = self.fields.height();
        let max_food = self.fields.channel_sum(FieldLayer::FoodPheromone).max(1.0);
        let mut out = format!(
            "P3\n# symtropy-colony tick {}\n{} {}\n255\n",
            self.tick, width, height
        );

        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = self.debug_pixel(x, y, max_food);
                out.push_str(&format!("{r} {g} {b} "));
            }
            out.push('\n');
        }

        out
    }

    fn debug_pixel(&self, x: usize, y: usize, max_food: f32) -> (u8, u8, u8) {
        if self.fields.is_obstacle(x, y) {
            return (12, 12, 12);
        }
        if (x, y) == self.nest {
            return (70, 130, 255);
        }
        if self.food_sources.get(&(x, y)).copied().unwrap_or(0) > 0 {
            return (40, 220, 80);
        }
        if self.ants.iter().any(|ant| ant.x == x && ant.y == y) {
            return (255, 240, 120);
        }

        let danger = self
            .fields
            .get(FieldLayer::DangerPheromone, x, y)
            .min(255.0) as u8;
        let food = self.fields.get(FieldLayer::FoodPheromone, x, y);
        let food_norm = ((food / max_food.sqrt()).sqrt() * 255.0).clamp(0.0, 255.0) as u8;
        let home = self.fields.get(FieldLayer::HomePheromone, x, y).min(255.0) as u8;

        (danger.max(food_norm), home / 3, food_norm)
    }

    fn reinforce_sources(&mut self) {
        let (nx, ny) = self.nest;
        self.fields.add(FieldLayer::HomePheromone, nx, ny, 30.0);
        for (&(x, y), &amount) in &self.food_sources {
            if amount > 0 {
                self.fields.add(FieldLayer::FoodPheromone, x, y, 35.0);
            }
        }
    }

    fn next_target(&self, ant_index: usize, width: usize, height: usize) -> (usize, usize) {
        let ant = &self.ants[ant_index];
        let options = self.neighbors(ant.x, ant.y, width, height);
        let mut best = (ant.x, ant.y);
        let mut best_score = f32::MIN;

        for (x, y) in options {
            if self.fields.is_obstacle(x, y) {
                continue;
            }

            let danger = self.fields.get(FieldLayer::DangerPheromone, x, y);
            let target_score = if ant.carrying_food {
                self.home_score(x, y)
            } else {
                self.food_score(x, y, ant.role, ant_index)
            };
            let score = target_score - danger * 4.0;
            if score > best_score {
                best_score = score;
                best = (x, y);
            }
        }

        best
    }

    fn food_score(&self, x: usize, y: usize, role: AntRole, ant_index: usize) -> f32 {
        let field = self.fields.get(FieldLayer::FoodPheromone, x, y);
        let mut score = field;
        if let Some(path) = self.shortest_path_distance((x, y), |p| {
            self.food_sources.get(&p).copied().unwrap_or(0) > 0
        }) {
            score += 160.0 / (path as f32 + 1.0);
        }
        if role == AntRole::Scout {
            score += self.deterministic_jitter(x, y, ant_index) * 0.1;
        }
        score
    }

    fn home_score(&self, x: usize, y: usize) -> f32 {
        let home_field = self.fields.get(FieldLayer::HomePheromone, x, y);
        let distance = x.abs_diff(self.nest.0) + y.abs_diff(self.nest.1);
        let path_bonus = self
            .shortest_path_distance((x, y), |p| p == self.nest)
            .map(|path| 180.0 / (path as f32 + 1.0))
            .unwrap_or(0.0);
        home_field + path_bonus + 20.0 / (distance as f32 + 1.0)
    }

    fn update_mind(&mut self) {
        let total = self.fields.channel_sum(FieldLayer::FoodPheromone);
        let active = self.count_active_food_cells();
        self.mind.trail_entropy = if total <= f32::EPSILON {
            1.0
        } else {
            (active as f32 / (self.fields.width() * self.fields.height()) as f32).min(1.0)
        };

        self.mind.pheromone_coherence = if total <= f32::EPSILON {
            0.0
        } else {
            self.strongest_food_corridor_sum() / total
        };

        self.mind.stress = self.fields.channel_sum(FieldLayer::DangerPheromone)
            / (self.fields.width() * self.fields.height()) as f32;
        self.mind.scout_ratio = if self.mind.trail_entropy > 0.35 || self.mind.stress > 0.5 {
            0.45
        } else {
            0.20
        };
    }

    fn count_active_food_cells(&self) -> usize {
        let mut count = 0;
        for y in 0..self.fields.height() {
            for x in 0..self.fields.width() {
                if self.fields.get(FieldLayer::FoodPheromone, x, y) > 1.0 {
                    count += 1;
                }
            }
        }
        count
    }

    fn strongest_food_corridor_sum(&self) -> f32 {
        let Some((&food_pos, _)) = self.food_sources.iter().find(|(_, amount)| **amount > 0) else {
            return 0.0;
        };
        let Some(path) = self.shortest_path(food_pos, self.nest) else {
            return 0.0;
        };
        path.into_iter()
            .map(|(x, y)| self.fields.get(FieldLayer::FoodPheromone, x, y))
            .sum()
    }

    fn shortest_path_distance<F>(&self, start: (usize, usize), goal: F) -> Option<usize>
    where
        F: Fn((usize, usize)) -> bool,
    {
        self.shortest_path_to(start, goal).map(|path| path.len())
    }

    fn shortest_path(
        &self,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<Vec<(usize, usize)>> {
        self.shortest_path_to(start, |p| p == goal)
    }

    fn shortest_path_to<F>(&self, start: (usize, usize), goal: F) -> Option<Vec<(usize, usize)>>
    where
        F: Fn((usize, usize)) -> bool,
    {
        let width = self.fields.width();
        let height = self.fields.height();
        let mut visited = vec![false; width * height];
        let mut prev: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut queue = VecDeque::from([start]);
        visited[self.fields.idx(start.0, start.1)] = true;

        while let Some(pos) = queue.pop_front() {
            if goal(pos) {
                let mut path = vec![pos];
                let mut cursor = pos;
                while let Some(&p) = prev.get(&cursor) {
                    path.push(p);
                    cursor = p;
                }
                path.reverse();
                return Some(path);
            }

            for next in self.neighbors(pos.0, pos.1, width, height) {
                if self.fields.is_obstacle(next.0, next.1) {
                    continue;
                }
                let idx = self.fields.idx(next.0, next.1);
                if !visited[idx] {
                    visited[idx] = true;
                    prev.insert(next, pos);
                    queue.push_back(next);
                }
            }
        }

        None
    }

    fn neighbors(&self, x: usize, y: usize, width: usize, height: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(5);
        out.push((x, y));
        if x > 0 {
            out.push((x - 1, y));
        }
        if x + 1 < width {
            out.push((x + 1, y));
        }
        if y > 0 {
            out.push((x, y - 1));
        }
        if y + 1 < height {
            out.push((x, y + 1));
        }
        out
    }

    fn deterministic_jitter(&self, x: usize, y: usize, ant_index: usize) -> f32 {
        let n = (x as u64 * 73_856_093)
            ^ (y as u64 * 19_349_663)
            ^ (ant_index as u64 * 83_492_791)
            ^ self.tick;
        (n % 1_000) as f32 / 1_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nest_discovers_food_forms_trail_and_increases_reserve() {
        let mut world = ColonyWorld::new(32, 16, (2, 8), 64);
        world.add_food_source(28, 8, 5_000);

        world.run_steps(180);

        assert!(world.mind.food_reserve > 20);
        assert!(world.fields.channel_sum(FieldLayer::FoodPheromone) > 100.0);
        assert!(world.mind.pheromone_coherence > 0.12);
    }

    #[test]
    fn broken_trail_increases_entropy_and_recovers_via_new_path() {
        let mut world = ColonyWorld::new(32, 16, (2, 8), 64);
        world.add_food_source(28, 8, 5_000);
        world.run_steps(120);
        let baseline_reserve = world.mind.food_reserve;

        for y in 1..15 {
            if y != 3 {
                world.set_obstacle(15, y, true);
            }
        }
        world.run_steps(120);

        assert!(world.mind.food_reserve > baseline_reserve);
        assert!(world.mind.scout_ratio >= 0.20);
        assert!(world.mind.pheromone_coherence > 0.08);
    }

    #[test]
    fn danger_pheromone_reroutes_foragers() {
        let mut world = ColonyWorld::new(32, 16, (2, 8), 64);
        world.add_food_source(28, 8, 5_000);
        for y in 0..16 {
            world.add_danger(14, y, 200.0);
            world.add_danger(15, y, 200.0);
            world.add_danger(16, y, 200.0);
        }

        world.run_steps(100);

        assert!(world.traffic_in_danger_region(14, 16) < 10);
    }

    #[test]
    fn ppm_heatmap_exports_visible_colony_state() {
        let mut world = ColonyWorld::new(8, 4, (1, 2), 8);
        world.add_food_source(6, 2, 100);
        world.set_obstacle(3, 1, true);
        world.add_danger(4, 1, 200.0);
        world.run_steps(4);

        let ppm = world.to_ppm_heatmap();

        assert!(ppm.starts_with("P3\n# symtropy-colony tick 4\n8 4\n255\n"));
        assert!(ppm.contains("70 130 255"));
        assert!(ppm.contains("40 220 80"));
        assert!(ppm.contains("12 12 12"));
    }

    #[test]
    fn debug_metrics_track_colony_state() {
        let mut world = ColonyWorld::new(16, 8, (1, 4), 16);
        world.add_food_source(14, 4, 500);
        world.run_steps(20);

        let metrics = world.debug_metrics();

        assert_eq!(metrics.tick, 20);
        assert_eq!(metrics.food_reserve, world.mind.food_reserve);
        assert!(metrics.pheromone_coherence >= 0.0);
    }

    #[test]
    fn basin_toxin_and_moisture_raise_colony_danger() {
        let mut world = ColonyWorld::new(8, 5, (1, 2), 8);
        let mut basin = FieldGrid::new(8, 5);
        basin.set(FieldLayer::Toxin, 4, 2, 100.0);
        basin.set(FieldLayer::Moisture, 4, 2, 60.0);

        world.absorb_basin_fields(&basin);

        assert!(world.fields.get(FieldLayer::DangerPheromone, 4, 2) > 7.0);
        assert_eq!(world.fields.get(FieldLayer::DangerPheromone, 1, 2), 0.0);
    }
}
