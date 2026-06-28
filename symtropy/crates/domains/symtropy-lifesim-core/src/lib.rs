// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared primitives for field-based living-system simulations.
//!
//! This crate is intentionally CPU-first. It gives domain crates a deterministic
//! reference model before WGSL compute kernels or ECS visualization are added.

use arrow_array::{Float32Array, UInt32Array};
use std::fmt;
use std::ops::Range;

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

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
    Heat,
    Light,
    Oxygen,
    Disease,
    SignalNoise,
    NullContamination,
}

impl FieldLayer {
    pub const COUNT: usize = 14;

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
            Self::Heat => 8,
            Self::Light => 9,
            Self::Oxygen => 10,
            Self::Disease => 11,
            Self::SignalNoise => 12,
            Self::NullContamination => 13,
        }
    }
}

/// Simple organism needs before any higher-order intelligence is applied.
#[derive(Debug, Clone, PartialEq)]
pub struct Metabolism {
    pub energy: f32,
    pub hydration: f32,
    pub oxygen_need: f32,
    pub heat_tolerance: Range<f32>,
    pub toxin_tolerance: f32,
    pub hunger_rate: f32,
    pub recovery_rate: f32,
}

impl Default for Metabolism {
    fn default() -> Self {
        Self {
            energy: 1.0,
            hydration: 1.0,
            oxygen_need: 0.25,
            heat_tolerance: 5.0..35.0,
            toxin_tolerance: 0.30,
            hunger_rate: 0.01,
            recovery_rate: 0.02,
        }
    }
}

impl Metabolism {
    pub fn stress_from_fields(&self, heat_c: f32, toxin: f32, oxygen: f32, moisture: f32) -> f32 {
        let heat_stress = if heat_c < self.heat_tolerance.start {
            (self.heat_tolerance.start - heat_c) / 20.0
        } else if heat_c > self.heat_tolerance.end {
            (heat_c - self.heat_tolerance.end) / 20.0
        } else {
            0.0
        };
        let toxin_stress = (toxin - self.toxin_tolerance).max(0.0);
        let oxygen_stress = (self.oxygen_need - oxygen).max(0.0);
        let hydration_stress = (self.hydration - moisture).max(0.0) * 0.25;
        sanitize_unit(heat_stress + toxin_stress + oxygen_stress + hydration_stress)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifeStage {
    Seed,
    Larva,
    Juvenile,
    Adult,
    Elder,
    Dormant,
    Dead,
    Decomposing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EcologicalRole {
    Forager,
    Decomposer,
    Pollinator,
    Grazer,
    Predator,
    Filterer,
    Engineer,
    Symbiont,
    Scout,
    Sentinel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Disposition {
    Thriving,
    Stressed,
    Disturbed,
    Defensive,
    Withdrawing,
    Hostile,
    Dormant,
    Collapsing,
}

impl Disposition {
    pub fn from_viability(viability: f32, stress: f32) -> Self {
        let viability = sanitize_unit(viability);
        let stress = sanitize_unit(stress);
        if viability >= 0.75 && stress < 0.20 {
            Self::Thriving
        } else if viability >= 0.55 && stress < 0.45 {
            Self::Stressed
        } else if viability >= 0.40 && stress < 0.65 {
            Self::Disturbed
        } else if viability >= 0.30 {
            Self::Withdrawing
        } else {
            Self::Collapsing
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifeSimEvent {
    TrailEstablished,
    TrailBroken,
    FoodSourceDepleted,
    ColonyStarving,
    ColonyRecovered,
    WetlandFiltering,
    ToxinSpike,
    BloomStarted,
    BloomCollapsed,
    SpeciesIntroduced,
    SpeciesBecameInvasive,
    EcologicalRepairSucceeded,
    EcologicalRepairBackfired,
    SignalCorruptionDetected,
    LivingInfrastructureDisturbed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TestimonyChannel {
    Scan,
    Diagnostic,
    Civic,
    Archive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LifeTestimony {
    pub channel: TestimonyChannel,
    pub summary: String,
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

    pub fn max_abs_diff(&self, other: &Self, layer: FieldLayer) -> f32 {
        assert_eq!(self.width, other.width, "field widths differ");
        assert_eq!(self.height, other.height, "field heights differ");
        self.channels[layer.index()]
            .iter()
            .zip(&other.channels[layer.index()])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
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

#[derive(Debug, Clone)]
pub struct FieldStepRequest {
    pub layer: FieldLayer,
    pub source: Vec<f32>,
    pub params: DiffusionParams,
}

impl FieldStepRequest {
    pub fn without_source(layer: FieldLayer, field: &FieldGrid, params: DiffusionParams) -> Self {
        Self {
            layer,
            source: vec![0.0; field.width() * field.height()],
            params,
        }
    }
}

pub trait FieldStepper {
    fn step(&self, field: &mut FieldGrid, request: &FieldStepRequest)
    -> Result<(), FieldStepError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFieldStepper;

impl FieldStepper for CpuFieldStepper {
    fn step(
        &self,
        field: &mut FieldGrid,
        request: &FieldStepRequest,
    ) -> Result<(), FieldStepError> {
        field.try_step_diffuse_decay_with_source(request.layer, &request.source, request.params)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldParityReport {
    pub max_abs_diff: f32,
    pub epsilon: f32,
}

impl FieldParityReport {
    pub const fn within_epsilon(&self) -> bool {
        self.max_abs_diff <= self.epsilon
    }
}

pub fn compare_layer_within_epsilon(
    expected: &FieldGrid,
    actual: &FieldGrid,
    layer: FieldLayer,
    epsilon: f32,
) -> FieldParityReport {
    FieldParityReport {
        max_abs_diff: expected.max_abs_diff(actual, layer),
        epsilon,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DiffusionParams {
    pub diffusion: f32,
    pub decay: f32,
    pub dt: f32,
    pub max_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldStepError {
    NonFiniteParam(&'static str),
    NegativeParam(&'static str),
    NonPositiveMaxValue,
    ExplicitDiffusionUnstable { courant: f32, max_courant: f32 },
    GpuUnavailable(String),
    GpuDispatchFailed(String),
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
            Self::GpuUnavailable(message) => write!(f, "GPU field stepper unavailable: {message}"),
            Self::GpuDispatchFailed(message) => write!(f, "GPU field dispatch failed: {message}"),
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

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn universal_ecology_layers_append_after_existing_layers() {
        assert_eq!(FieldLayer::FoodPheromone.index(), 0);
        assert_eq!(FieldLayer::Biomass.index(), 7);
        assert_eq!(FieldLayer::Heat.index(), 8);
        assert_eq!(FieldLayer::NullContamination.index(), FieldLayer::COUNT - 1);

        let mut field = FieldGrid::new(2, 2);
        field.set(FieldLayer::Oxygen, 1, 1, 0.72);
        field.set(FieldLayer::SignalNoise, 0, 1, 0.31);

        assert_eq!(field.get(FieldLayer::Oxygen, 1, 1), 0.72);
        assert_eq!(field.get(FieldLayer::SignalNoise, 0, 1), 0.31);
    }

    #[test]
    fn metabolism_reports_stress_from_environmental_fields() {
        let metabolism = Metabolism {
            toxin_tolerance: 0.2,
            oxygen_need: 0.4,
            heat_tolerance: 10.0..30.0,
            ..Metabolism::default()
        };

        let calm = metabolism.stress_from_fields(20.0, 0.1, 0.8, 1.0);
        let stressed = metabolism.stress_from_fields(44.0, 0.8, 0.1, 0.2);

        assert_eq!(calm, 0.0);
        assert!(stressed > 0.8);
    }

    #[test]
    fn disposition_tracks_viability_and_stress_without_combat_identity() {
        assert_eq!(
            Disposition::from_viability(0.90, 0.05),
            Disposition::Thriving
        );
        assert_eq!(
            Disposition::from_viability(0.45, 0.60),
            Disposition::Disturbed
        );
        assert_eq!(
            Disposition::from_viability(0.20, 0.90),
            Disposition::Collapsing
        );
    }

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
    fn cpu_stepper_matches_direct_reference_step() {
        let mut direct = FieldGrid::new(5, 5);
        let mut stepped = FieldGrid::new(5, 5);
        direct.set(FieldLayer::Moisture, 2, 2, 20.0);
        stepped.set(FieldLayer::Moisture, 2, 2, 20.0);

        let params = DiffusionParams::default();
        let request = FieldStepRequest::without_source(FieldLayer::Moisture, &stepped, params);
        CpuFieldStepper.step(&mut stepped, &request).unwrap();
        direct.step_diffuse_decay(FieldLayer::Moisture, params);

        let report = compare_layer_within_epsilon(&direct, &stepped, FieldLayer::Moisture, 0.0);
        assert!(report.within_epsilon(), "report={report:?}");
    }

    #[test]
    fn arrow_snapshot_schema_signature_is_stable() {
        let field = FieldGrid::new(2, 2);
        let snapshot = field.to_arrow_snapshot(FieldLayer::FoodPheromone);

        assert_eq!(snapshot.schema_signature(), "x:u32,y:u32,value:f32");
    }
}
