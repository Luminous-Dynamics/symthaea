// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared primitives for field-based living-system simulations.
//!
//! This crate is intentionally CPU-first. It gives domain crates a deterministic
//! reference model before WGSL compute kernels or ECS visualization are added.

use arrow_array::{Float32Array, UInt32Array};
use std::fmt;

/// Canonical field channels shared by ant colonies, mycelium, wetlands, and biofilms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldLayer {
    FoodPheromone,
    HomePheromone,
    DangerPheromone,
    Moisture,
    Obstacle,
    Nutrient,
    Toxin,
    Biomass,
}

impl FieldLayer {
    pub const COUNT: usize = 8;

    pub const fn index(self) -> usize {
        match self {
            Self::FoodPheromone => 0,
            Self::HomePheromone => 1,
            Self::DangerPheromone => 2,
            Self::Moisture => 3,
            Self::Obstacle => 4,
            Self::Nutrient => 5,
            Self::Toxin => 6,
            Self::Biomass => 7,
        }
    }
}

/// Dense 2D multi-channel field grid.
#[derive(Debug, Clone)]
pub struct FieldGrid {
    width: usize,
    height: usize,
    channels: Vec<Vec<f32>>,
}

impl FieldGrid {
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0, "field width must be non-zero");
        assert!(height > 0, "field height must be non-zero");
        let cells = width * height;
        Self {
            width,
            height,
            channels: vec![vec![0.0; cells]; FieldLayer::COUNT],
        }
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn idx(&self, x: usize, y: usize) -> usize {
        assert!(x < self.width, "x out of bounds");
        assert!(y < self.height, "y out of bounds");
        y * self.width + x
    }

    pub fn get(&self, layer: FieldLayer, x: usize, y: usize) -> f32 {
        self.channels[layer.index()][self.idx(x, y)]
    }

    pub fn set(&mut self, layer: FieldLayer, x: usize, y: usize, value: f32) {
        let idx = self.idx(x, y);
        self.channels[layer.index()][idx] = sanitize_concentration(value, f32::MAX);
    }

    pub fn add(&mut self, layer: FieldLayer, x: usize, y: usize, value: f32) {
        let idx = self.idx(x, y);
        let channel = &mut self.channels[layer.index()];
        channel[idx] = sanitize_concentration(channel[idx] + value, f32::MAX);
    }

    pub fn is_obstacle(&self, x: usize, y: usize) -> bool {
        self.get(FieldLayer::Obstacle, x, y) >= 0.5
    }

    pub fn channel_sum(&self, layer: FieldLayer) -> f32 {
        self.channels[layer.index()].iter().sum()
    }

    pub fn step_diffuse_decay(&mut self, layer: FieldLayer, params: DiffusionParams) {
        let source = vec![0.0; self.width * self.height];
        self.try_step_diffuse_decay_with_source(layer, &source, params)
            .expect("invalid diffusion parameters");
    }

    pub fn step_diffuse_decay_with_source(
        &mut self,
        layer: FieldLayer,
        source: &[f32],
        params: DiffusionParams,
    ) {
        self.try_step_diffuse_decay_with_source(layer, source, params)
            .expect("invalid diffusion parameters");
    }

    pub fn try_step_diffuse_decay(
        &mut self,
        layer: FieldLayer,
        params: DiffusionParams,
    ) -> Result<(), FieldStepError> {
        let source = vec![0.0; self.width * self.height];
        self.try_step_diffuse_decay_with_source(layer, &source, params)
    }

    pub fn try_step_diffuse_decay_with_source(
        &mut self,
        layer: FieldLayer,
        source: &[f32],
        params: DiffusionParams,
    ) -> Result<(), FieldStepError> {
        params.validate()?;
        assert_eq!(source.len(), self.width * self.height);
        let input = self.channels[layer.index()].clone();
        let mut output = input.clone();

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.idx(x, y);
                if self.is_obstacle(x, y) {
                    output[idx] = 0.0;
                    continue;
                }

                let center = input[idx];
                let left = self.diffusion_neighbor(&input, x.checked_sub(1), Some(y), center);
                let right = self.diffusion_neighbor(&input, x.checked_add(1), Some(y), center);
                let up = self.diffusion_neighbor(&input, Some(x), y.checked_sub(1), center);
                let down = self.diffusion_neighbor(&input, Some(x), y.checked_add(1), center);
                let laplacian = left + right + up + down - 4.0 * center;
                let source_value = sanitize_source(source[idx]);
                let next = center
                    + params.dt
                        * (params.diffusion * laplacian - params.decay * center + source_value);
                output[idx] = sanitize_concentration(next, params.max_value);
            }
        }

        self.channels[layer.index()] = output;
        Ok(())
    }

    fn diffusion_neighbor(
        &self,
        input: &[f32],
        x: Option<usize>,
        y: Option<usize>,
        fallback: f32,
    ) -> f32 {
        let (Some(x), Some(y)) = (x, y) else {
            return fallback;
        };
        if x >= self.width || y >= self.height || self.is_obstacle(x, y) {
            return fallback;
        }
        input[self.idx(x, y)]
    }

    /// Export one field layer as Arrow arrays for telemetry/replay pipelines.
    pub fn to_arrow_snapshot(&self, layer: FieldLayer) -> FieldSnapshotArrays {
        let mut xs = Vec::with_capacity(self.width * self.height);
        let mut ys = Vec::with_capacity(self.width * self.height);
        let mut values = Vec::with_capacity(self.width * self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                xs.push(x as u32);
                ys.push(y as u32);
                values.push(self.get(layer, x, y));
            }
        }

        FieldSnapshotArrays {
            x: UInt32Array::from(xs),
            y: UInt32Array::from(ys),
            value: Float32Array::from(values),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DiffusionParams {
    pub diffusion: f32,
    pub decay: f32,
    pub dt: f32,
    pub max_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldStepError {
    NonFiniteParam(&'static str),
    NegativeParam(&'static str),
    NonPositiveMaxValue,
    ExplicitDiffusionUnstable { courant: f32, max_courant: f32 },
}

impl fmt::Display for FieldStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteParam(name) => write!(f, "{name} must be finite"),
            Self::NegativeParam(name) => write!(f, "{name} must be non-negative"),
            Self::NonPositiveMaxValue => write!(f, "max_value must be positive"),
            Self::ExplicitDiffusionUnstable {
                courant,
                max_courant,
            } => write!(
                f,
                "explicit 2D diffusion is unstable: diffusion * dt = {courant}, max {max_courant}"
            ),
        }
    }
}

impl std::error::Error for FieldStepError {}

impl DiffusionParams {
    pub const MAX_EXPLICIT_2D_COURANT: f32 = 0.25;

    pub fn validate(self) -> Result<(), FieldStepError> {
        for (name, value) in [
            ("diffusion", self.diffusion),
            ("decay", self.decay),
            ("dt", self.dt),
            ("max_value", self.max_value),
        ] {
            if !value.is_finite() {
                return Err(FieldStepError::NonFiniteParam(name));
            }
        }

        for (name, value) in [
            ("diffusion", self.diffusion),
            ("decay", self.decay),
            ("dt", self.dt),
        ] {
            if value < 0.0 {
                return Err(FieldStepError::NegativeParam(name));
            }
        }

        if self.max_value <= 0.0 {
            return Err(FieldStepError::NonPositiveMaxValue);
        }

        let courant = self.diffusion * self.dt;
        if courant > Self::MAX_EXPLICIT_2D_COURANT {
            return Err(FieldStepError::ExplicitDiffusionUnstable {
                courant,
                max_courant: Self::MAX_EXPLICIT_2D_COURANT,
            });
        }

        Ok(())
    }
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            diffusion: 0.15,
            decay: 0.01,
            dt: 1.0,
            max_value: 1_000.0,
        }
    }
}

pub struct FieldSnapshotArrays {
    pub x: UInt32Array,
    pub y: UInt32Array,
    pub value: Float32Array,
}

impl FieldSnapshotArrays {
    pub const SCHEMA_SIGNATURE: &'static str = "x:u32,y:u32,value:f32";

    pub const fn schema_signature(&self) -> &'static str {
        Self::SCHEMA_SIGNATURE
    }
}

fn sanitize_concentration(value: f32, max_value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, max_value)
}

fn sanitize_source(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusion_preserves_source_shape_and_decays() {
        let mut field = FieldGrid::new(5, 5);
        field.set(FieldLayer::FoodPheromone, 2, 2, 10.0);
        field.step_diffuse_decay(FieldLayer::FoodPheromone, DiffusionParams::default());

        assert!(field.get(FieldLayer::FoodPheromone, 2, 2) < 10.0);
        assert!(field.get(FieldLayer::FoodPheromone, 1, 2) > 0.0);
    }

    #[test]
    fn arrow_snapshot_has_one_row_per_cell() {
        let mut field = FieldGrid::new(3, 2);
        field.set(FieldLayer::Moisture, 1, 1, 0.7);
        let snapshot = field.to_arrow_snapshot(FieldLayer::Moisture);

        assert_eq!(snapshot.x.len(), 6);
        assert_eq!(snapshot.y.len(), 6);
        assert_eq!(snapshot.value.len(), 6);
        assert_eq!(snapshot.value.value(field.idx(1, 1)), 0.7);
    }

    #[test]
    fn rejects_unstable_explicit_diffusion_params() {
        let mut field = FieldGrid::new(3, 3);
        let result = field.try_step_diffuse_decay(
            FieldLayer::Moisture,
            DiffusionParams {
                diffusion: 0.5,
                dt: 1.0,
                ..DiffusionParams::default()
            },
        );

        assert!(matches!(
            result,
            Err(FieldStepError::ExplicitDiffusionUnstable { .. })
        ));
    }

    #[test]
    fn diffusion_never_produces_nan_or_negative_values() {
        let mut field = FieldGrid::new(5, 5);
        field.set(FieldLayer::Nutrient, 2, 2, f32::NAN);
        field.add(FieldLayer::Nutrient, 2, 2, 10.0);
        let mut source = vec![0.0; 25];
        source[field.idx(1, 1)] = f32::NAN;
        source[field.idx(3, 3)] = -100.0;

        field
            .try_step_diffuse_decay_with_source(
                FieldLayer::Nutrient,
                &source,
                DiffusionParams::default(),
            )
            .unwrap();

        for y in 0..field.height() {
            for x in 0..field.width() {
                let value = field.get(FieldLayer::Nutrient, x, y);
                assert!(value.is_finite());
                assert!(value >= 0.0);
            }
        }
    }

    #[test]
    fn source_injection_increases_local_mass() {
        let mut field = FieldGrid::new(5, 5);
        let before = field.channel_sum(FieldLayer::Biomass);
        let mut source = vec![0.0; 25];
        source[field.idx(2, 2)] = 5.0;

        field
            .try_step_diffuse_decay_with_source(
                FieldLayer::Biomass,
                &source,
                DiffusionParams {
                    diffusion: 0.0,
                    decay: 0.0,
                    dt: 1.0,
                    max_value: 1_000.0,
                },
            )
            .unwrap();

        assert!(field.channel_sum(FieldLayer::Biomass) > before);
        assert_eq!(field.get(FieldLayer::Biomass, 2, 2), 5.0);
    }

    #[test]
    fn obstacles_block_cross_cell_diffusion() {
        let mut field = FieldGrid::new(5, 3);
        field.set(FieldLayer::FoodPheromone, 1, 1, 10.0);
        field.set(FieldLayer::Obstacle, 2, 1, 1.0);

        field.step_diffuse_decay(FieldLayer::FoodPheromone, DiffusionParams::default());

        assert_eq!(field.get(FieldLayer::FoodPheromone, 2, 1), 0.0);
        assert_eq!(field.get(FieldLayer::FoodPheromone, 3, 1), 0.0);
    }

    #[test]
    fn cpu_diffusion_is_deterministic() {
        let mut a = FieldGrid::new(6, 6);
        let mut b = FieldGrid::new(6, 6);
        a.set(FieldLayer::Toxin, 3, 3, 12.0);
        b.set(FieldLayer::Toxin, 3, 3, 12.0);

        for _ in 0..8 {
            a.step_diffuse_decay(FieldLayer::Toxin, DiffusionParams::default());
            b.step_diffuse_decay(FieldLayer::Toxin, DiffusionParams::default());
        }

        for y in 0..a.height() {
            for x in 0..a.width() {
                assert_eq!(
                    a.get(FieldLayer::Toxin, x, y),
                    b.get(FieldLayer::Toxin, x, y)
                );
            }
        }
    }

    #[test]
    fn arrow_snapshot_schema_signature_is_stable() {
        let field = FieldGrid::new(2, 2);
        let snapshot = field.to_arrow_snapshot(FieldLayer::FoodPheromone);

        assert_eq!(snapshot.schema_signature(), "x:u32,y:u32,value:f32");
    }
}
