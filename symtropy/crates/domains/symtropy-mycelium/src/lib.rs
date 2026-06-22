// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mycelial Network V0: field-based growth, transport, and toxin avoidance.
//!
//! This crate is deliberately CPU-first. It proves that `symtropy-lifesim-core`
//! is not ant-specific by reusing the same dense field substrate for a sessile,
//! graph-like living system.

use std::collections::{HashMap, VecDeque};

use symtropy_lifesim_core::{DiffusionParams, FieldGrid, FieldLayer};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MyceliumMetrics {
    pub tick: u64,
    pub central_biomass: f32,
    pub total_biomass: f32,
    pub total_nutrient: f32,
    pub total_toxin: f32,
}

#[derive(Debug, Clone)]
pub struct MyceliumWorld {
    pub fields: FieldGrid,
    pub origin: (usize, usize),
    tick: u64,
}

impl MyceliumWorld {
    pub fn new(width: usize, height: usize, origin: (usize, usize)) -> Self {
        let mut fields = FieldGrid::new(width, height);
        fields.set(FieldLayer::Biomass, origin.0, origin.1, 8.0);
        Self {
            fields,
            origin,
            tick: 0,
        }
    }

    pub const fn tick(&self) -> u64 {
        self.tick
    }

    pub fn add_dead_biomass(&mut self, x: usize, y: usize, nutrients: f32) {
        self.fields.add(FieldLayer::Nutrient, x, y, nutrients);
    }

    pub fn add_toxin(&mut self, x: usize, y: usize, intensity: f32) {
        self.fields.add(FieldLayer::Toxin, x, y, intensity);
    }

    pub fn biomass_at(&self, x: usize, y: usize) -> f32 {
        self.fields.get(FieldLayer::Biomass, x, y)
    }

    pub fn metrics(&self) -> MyceliumMetrics {
        MyceliumMetrics {
            tick: self.tick,
            central_biomass: self.biomass_at(self.origin.0, self.origin.1),
            total_biomass: self.fields.channel_sum(FieldLayer::Biomass),
            total_nutrient: self.fields.channel_sum(FieldLayer::Nutrient),
            total_toxin: self.fields.channel_sum(FieldLayer::Toxin),
        }
    }

    pub fn run_steps(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    pub fn step(&mut self) {
        self.tick += 1;
        self.fields.step_diffuse_decay(
            FieldLayer::Nutrient,
            DiffusionParams {
                diffusion: 0.12,
                decay: 0.005,
                dt: 1.0,
                max_value: 1_000.0,
            },
        );
        self.fields.step_diffuse_decay(
            FieldLayer::Toxin,
            DiffusionParams {
                diffusion: 0.05,
                decay: 0.01,
                dt: 1.0,
                max_value: 1_000.0,
            },
        );

        let mut biomass_delta = HashMap::<(usize, usize), f32>::new();
        let mut nutrient_delta = HashMap::<(usize, usize), f32>::new();
        let mut transported_to_origin = 0.0;

        for y in 0..self.fields.height() {
            for x in 0..self.fields.width() {
                let biomass = self.fields.get(FieldLayer::Biomass, x, y);
                if biomass < 0.25 {
                    continue;
                }

                let local_nutrient = self.fields.get(FieldLayer::Nutrient, x, y);
                let local_toxin = self.fields.get(FieldLayer::Toxin, x, y);
                let assimilation = (local_nutrient * 0.08).min(biomass * 0.04).max(0.0);
                if local_toxin < 5.0 && assimilation > 0.0 {
                    *biomass_delta.entry((x, y)).or_default() += assimilation;
                    *nutrient_delta.entry((x, y)).or_default() -= assimilation;
                    transported_to_origin += assimilation * 0.25;
                }

                if biomass < 0.05 {
                    continue;
                }

                let nutrient_peak = self.strongest_nutrient_cell();
                let Some(target) = self.best_growth_neighbor(x, y, nutrient_peak) else {
                    continue;
                };
                let toxin = self.fields.get(FieldLayer::Toxin, target.0, target.1);
                if toxin >= 20.0 {
                    continue;
                }
                let nutrient = self.fields.get(FieldLayer::Nutrient, target.0, target.1);
                let growth = (0.08 + nutrient * 0.06).min(biomass * 0.25);
                if growth > 0.0 {
                    *biomass_delta.entry(target).or_default() += growth;
                    *biomass_delta.entry((x, y)).or_default() -= growth * 0.05;
                    *nutrient_delta.entry(target).or_default() -= growth * 0.35;
                }
            }
        }

        for ((x, y), delta) in biomass_delta {
            self.fields.add(FieldLayer::Biomass, x, y, delta);
        }
        for ((x, y), delta) in nutrient_delta {
            self.fields.add(FieldLayer::Nutrient, x, y, delta);
        }
        self.fields.add(
            FieldLayer::Biomass,
            self.origin.0,
            self.origin.1,
            transported_to_origin,
        );

        self.fields.step_diffuse_decay(
            FieldLayer::Biomass,
            DiffusionParams {
                diffusion: 0.0,
                decay: 0.002,
                dt: 1.0,
                max_value: 1_000.0,
            },
        );
    }

    pub fn has_biomass_path_to(&self, goal: (usize, usize), threshold: f32) -> bool {
        if self.biomass_at(self.origin.0, self.origin.1) < threshold {
            return false;
        }
        let mut visited = vec![false; self.fields.width() * self.fields.height()];
        let mut queue = VecDeque::from([self.origin]);
        visited[self.fields.idx(self.origin.0, self.origin.1)] = true;

        while let Some(pos) = queue.pop_front() {
            if pos == goal {
                return true;
            }
            for next in self.neighbors(pos.0, pos.1) {
                let idx = self.fields.idx(next.0, next.1);
                if visited[idx] || self.biomass_at(next.0, next.1) < threshold {
                    continue;
                }
                visited[idx] = true;
                queue.push_back(next);
            }
        }

        false
    }

    fn best_growth_neighbor(
        &self,
        x: usize,
        y: usize,
        nutrient_peak: Option<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        self.neighbors(x, y).into_iter().max_by(|a, b| {
            self.growth_score(*a, nutrient_peak)
                .partial_cmp(&self.growth_score(*b, nutrient_peak))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn growth_score(&self, pos: (usize, usize), nutrient_peak: Option<(usize, usize)>) -> f32 {
        let nutrient = self.fields.get(FieldLayer::Nutrient, pos.0, pos.1);
        let toxin = self.fields.get(FieldLayer::Toxin, pos.0, pos.1);
        let existing = self.fields.get(FieldLayer::Biomass, pos.0, pos.1);
        let peak_pull = nutrient_peak
            .map(|peak| {
                let distance = pos.0.abs_diff(peak.0) + pos.1.abs_diff(peak.1);
                12.0 / (distance as f32 + 1.0)
            })
            .unwrap_or(0.0);
        nutrient * 2.0 + peak_pull - existing * 0.05 - toxin * 4.0
    }

    fn strongest_nutrient_cell(&self) -> Option<(usize, usize)> {
        let mut best = None;
        let mut best_value = 0.0;
        for y in 0..self.fields.height() {
            for x in 0..self.fields.width() {
                let value = self.fields.get(FieldLayer::Nutrient, x, y);
                if value > best_value {
                    best_value = value;
                    best = Some((x, y));
                }
            }
        }
        best
    }

    fn neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(4);
        if x > 0 {
            out.push((x - 1, y));
        }
        if x + 1 < self.fields.width() {
            out.push((x + 1, y));
        }
        if y > 0 {
            out.push((x, y - 1));
        }
        if y + 1 < self.fields.height() {
            out.push((x, y + 1));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_follows_nutrient_gradient() {
        let mut world = MyceliumWorld::new(16, 9, (3, 4));
        world.add_dead_biomass(13, 4, 400.0);

        world.run_steps(80);

        let east_biomass: f32 = (8..16).map(|x| world.biomass_at(x, 4)).sum();
        let west_biomass: f32 = (0..3).map(|x| world.biomass_at(x, 4)).sum();
        assert!(
            east_biomass > west_biomass,
            "east_biomass={east_biomass}, west_biomass={west_biomass}"
        );
    }

    #[test]
    fn toxin_suppresses_expansion() {
        let mut clean = MyceliumWorld::new(16, 9, (3, 4));
        clean.add_dead_biomass(13, 4, 400.0);
        clean.run_steps(80);

        let mut toxic = MyceliumWorld::new(16, 9, (3, 4));
        toxic.add_dead_biomass(13, 4, 400.0);
        for x in 7..11 {
            for y in 2..7 {
                toxic.add_toxin(x, y, 200.0);
            }
        }
        toxic.run_steps(80);

        let clean_far_biomass: f32 = (11..16).map(|x| clean.biomass_at(x, 4)).sum();
        let toxic_far_biomass: f32 = (11..16).map(|x| toxic.biomass_at(x, 4)).sum();
        assert!(
            toxic_far_biomass < clean_far_biomass * 0.5,
            "toxic_far_biomass={toxic_far_biomass}, clean_far_biomass={clean_far_biomass}"
        );
    }

    #[test]
    fn network_preserves_connected_path_to_food() {
        let mut world = MyceliumWorld::new(16, 9, (3, 4));
        world.add_dead_biomass(13, 4, 800.0);

        world.run_steps(120);

        assert!(
            world.has_biomass_path_to((13, 4), 0.08),
            "metrics={:?}, target_biomass={}",
            world.metrics(),
            world.biomass_at(13, 4)
        );
    }

    #[test]
    fn nutrient_transport_increases_central_biomass() {
        let mut world = MyceliumWorld::new(16, 9, (3, 4));
        let before = world.metrics().central_biomass;
        world.add_dead_biomass(10, 4, 600.0);

        world.run_steps(100);

        assert!(world.metrics().central_biomass > before);
    }
}
