// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strong-simple architecture baselines for SYM-ARCH-002.
//!
//! This module deliberately starts with models that are simpler than Symthaea's
//! liquid/Hebbian architecture. The central comparison is readout-matched:
//!
//! 1. categorical one-hot features + online recursive least squares (RLS),
//! 2. fixed nonlinear random features + the same RLS readout, and
//! 3. vanilla HDC role/value binding + the same RLS readout.
//!
//! None of these baselines has temporal state, replay, or learned encoder
//! parameters. The full RLS covariance is adaptive state, however, and is
//! reported explicitly because replay-free does not mean memory-free.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_core::hdc::ContinuousHV;

pub const SIMPLE_BASELINE_SCHEMA_V1: &str = "symthaea.simple-baseline/v1";
/// Hard safety ceiling for a full f64 RLS inverse-covariance matrix.
///
/// This is not a model-selection knob. It prevents accidental multi-gigabyte
/// allocation when a caller tries to pair full-covariance RLS with Symthaea's
/// normal 16K+ HDC dimensions. Larger comparisons need a different analytic
/// readout (diagonal/low-rank/block) rather than silently exhausting memory.
pub const MAX_RLS_COVARIANCE_BYTES: usize = 512 * 1024 * 1024;

const BASELINE_SPEC_HASH_DOMAIN: &[u8] = b"symthaea.simple-baseline.spec.hash/v1";
const RANDOM_FEATURE_DOMAIN: &[u8] = b"symthaea.simple-baseline.random-feature/v1";
const HDC_ROLE_DOMAIN: &[u8] = b"symthaea.simple-baseline.hdc-role/v1";
const HDC_VALUE_DOMAIN: &[u8] = b"symthaea.simple-baseline.hdc-value/v1";

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn derived_seed(seed: u64, domain: &[u8], payload: &[u8]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&[0]);
    hasher.update(payload);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("eight digest bytes"))
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn uniform_signed(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    let unit = bits as f64 * (1.0 / ((1u64 << 53) as f64));
    unit * 2.0 - 1.0
}

fn l2_normalize(values: &mut [f64]) -> Result<(), String> {
    let norm_sq: f64 = values.iter().map(|value| value * value).sum();
    if !norm_sq.is_finite() || norm_sq <= 1e-24 {
        return Err("encoded feature vector has zero/non-finite norm".into());
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for value in values {
        *value *= inv_norm;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoricalFeatureSpec {
    pub name: String,
    /// Frozen categorical support. Values must be strictly increasing.
    pub values: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoricalFeatureSchema {
    /// Frozen feature order. Names must be strictly lexicographically increasing.
    pub features: Vec<CategoricalFeatureSpec>,
}

impl CategoricalFeatureSchema {
    pub fn validate(&self) -> Result<(), String> {
        if self.features.is_empty() {
            return Err("categorical schema must contain at least one feature".into());
        }
        let mut previous_name: Option<&str> = None;
        for feature in &self.features {
            let name = feature.name.trim();
            if name.is_empty() || name != feature.name {
                return Err("feature names must be non-empty and already normalized".into());
            }
            if let Some(previous) = previous_name {
                if previous >= name {
                    return Err("feature names must be unique and strictly sorted".into());
                }
            }
            previous_name = Some(name);
            if feature.values.is_empty() {
                return Err(format!("feature {} has an empty domain", feature.name));
            }
            if feature.values.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(format!(
                    "feature {} values must be unique and strictly increasing",
                    feature.name
                ));
            }
        }
        Ok(())
    }

    pub fn one_hot_dimension(&self) -> Result<usize, String> {
        self.validate()?;
        self.features.iter().try_fold(0usize, |total, feature| {
            total
                .checked_add(feature.values.len())
                .ok_or_else(|| "one-hot dimension overflow".to_string())
        })
    }

    pub fn unique_values(&self) -> Result<Vec<i64>, String> {
        self.validate()?;
        let mut values = BTreeSet::new();
        for feature in &self.features {
            values.extend(feature.values.iter().copied());
        }
        Ok(values.into_iter().collect())
    }

    fn active_one_hot_indices(
        &self,
        assignment: &BTreeMap<String, i64>,
    ) -> Result<Vec<usize>, String> {
        self.validate()?;
        if assignment.len() != self.features.len() {
            return Err("assignment feature count does not match frozen schema".into());
        }
        let expected_names: BTreeSet<&str> = self.features.iter().map(|f| f.name.as_str()).collect();
        let observed_names: BTreeSet<&str> = assignment.keys().map(String::as_str).collect();
        if expected_names != observed_names {
            return Err("assignment feature names do not match frozen schema".into());
        }

        let mut indices = Vec::with_capacity(self.features.len());
        let mut offset = 0usize;
        for feature in &self.features {
            let value = assignment
                .get(&feature.name)
                .copied()
                .ok_or_else(|| format!("assignment missing feature {}", feature.name))?;
            let local = feature
                .values
                .binary_search(&value)
                .map_err(|_| format!("feature {} value {value} is out of schema", feature.name))?;
            indices.push(offset + local);
            offset += feature.values.len();
        }
        Ok(indices)
    }

    pub fn normalized_one_hot(
        &self,
        assignment: &BTreeMap<String, i64>,
    ) -> Result<Vec<f64>, String> {
        let dimension = self.one_hot_dimension()?;
        let active = self.active_one_hot_indices(assignment)?;
        let mut encoded = vec![0.0; dimension];
        let amplitude = 1.0 / (active.len() as f64).sqrt();
        for index in active {
            encoded[index] = amplitude;
        }
        Ok(encoded)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlsConfig {
    /// Ridge precision. The initial inverse covariance is I / ridge.
    pub ridge: f64,
    /// Standard RLS forgetting factor in (0, 1]. Use 1.0 for no recency decay.
    pub forgetting_factor: f64,
    pub include_bias: bool,
}

impl Default for RlsConfig {
    fn default() -> Self {
        Self {
            ridge: 1.0,
            forgetting_factor: 1.0,
            include_bias: true,
        }
    }
}

impl RlsConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.ridge.is_finite() || self.ridge <= 0.0 {
            return Err("RLS ridge must be finite and positive".into());
        }
        if !self.forgetting_factor.is_finite()
            || self.forgetting_factor <= 0.0
            || self.forgetting_factor > 1.0
        {
            return Err("RLS forgetting factor must be in (0, 1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OnlineRlsBinary {
    input_dimension: usize,
    state_dimension: usize,
    config: RlsConfig,
    weights: Vec<f64>,
    inverse_covariance: Vec<f64>,
    updates: usize,
}

impl OnlineRlsBinary {
    pub fn new(input_dimension: usize, config: RlsConfig) -> Result<Self, String> {
        config.validate()?;
        if input_dimension == 0 {
            return Err("RLS input dimension must be positive".into());
        }
        let state_dimension = input_dimension
            .checked_add(usize::from(config.include_bias))
            .ok_or_else(|| "RLS state dimension overflow".to_string())?;
        let covariance_len = state_dimension
            .checked_mul(state_dimension)
            .ok_or_else(|| "RLS covariance allocation overflow".to_string())?;
        let covariance_bytes = covariance_len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| "RLS covariance byte count overflow".to_string())?;
        if covariance_bytes > MAX_RLS_COVARIANCE_BYTES {
            return Err(format!(
                "full RLS covariance would require {covariance_bytes} bytes, above hard ceiling {MAX_RLS_COVARIANCE_BYTES}; use a smaller encoded dimension or a bounded-state analytic readout"
            ));
        }
        let mut inverse_covariance = vec![0.0; covariance_len];
        let diagonal = 1.0 / config.ridge;
        for index in 0..state_dimension {
            inverse_covariance[index * state_dimension + index] = diagonal;
        }
        Ok(Self {
            input_dimension,
            state_dimension,
            config,
            weights: vec![0.0; state_dimension],
            inverse_covariance,
            updates: 0,
        })
    }

    fn augment(&self, features: &[f64]) -> Result<Vec<f64>, String> {
        if features.len() != self.input_dimension {
            return Err(format!(
                "RLS feature dimension mismatch: expected {}, got {}",
                self.input_dimension,
                features.len()
            ));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err("RLS features must be finite".into());
        }
        let mut augmented = Vec::with_capacity(self.state_dimension);
        augmented.extend_from_slice(features);
        if self.config.include_bias {
            augmented.push(1.0);
        }
        Ok(augmented)
    }

    pub fn score(&self, features: &[f64]) -> Result<f64, String> {
        let x = self.augment(features)?;
        Ok(self.weights.iter().zip(x).map(|(weight, value)| weight * value).sum())
    }

    pub fn predict(&self, features: &[f64]) -> Result<bool, String> {
        Ok(self.score(features)? > 0.0)
    }

    pub fn update(&mut self, features: &[f64], label: bool) -> Result<f64, String> {
        let x = self.augment(features)?;
        let n = self.state_dimension;
        let mut px = vec![0.0; n];
        for row in 0..n {
            let base = row * n;
            let mut total = 0.0;
            for col in 0..n {
                total += self.inverse_covariance[base + col] * x[col];
            }
            px[row] = total;
        }
        let x_t_px: f64 = x.iter().zip(&px).map(|(left, right)| left * right).sum();
        let denominator = self.config.forgetting_factor + x_t_px;
        if !denominator.is_finite() || denominator <= 1e-15 {
            return Err("RLS update denominator became non-positive/non-finite".into());
        }

        let prediction: f64 = self.weights.iter().zip(&x).map(|(w, value)| w * value).sum();
        let target = if label { 1.0 } else { -1.0 };
        let error = target - prediction;
        for index in 0..n {
            self.weights[index] += (px[index] / denominator) * error;
        }

        // Because P is initialized symmetric, x^T P == (P x)^T. Updating by
        // the symmetric outer product Px(Px)^T avoids an unnecessary second
        // matrix-vector product and preserves symmetry up to roundoff.
        for row in 0..n {
            for col in 0..n {
                let index = row * n + col;
                self.inverse_covariance[index] =
                    (self.inverse_covariance[index] - px[row] * px[col] / denominator)
                        / self.config.forgetting_factor;
            }
        }
        if self
            .weights
            .iter()
            .chain(&self.inverse_covariance)
            .any(|value| !value.is_finite())
        {
            return Err("RLS state became non-finite".into());
        }
        self.updates += 1;
        Ok(error)
    }

    pub fn updates(&self) -> usize {
        self.updates
    }

    pub fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    pub fn trainable_parameters(&self) -> usize {
        self.state_dimension
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.len() * std::mem::size_of::<f64>()
    }

    pub fn covariance_bytes(&self) -> usize {
        self.inverse_covariance.len() * std::mem::size_of::<f64>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimpleBaselineKind {
    OneHotRls,
    FixedRandomTanhRls,
    VanillaHdcRls,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimpleBaselineSpec {
    pub schema: String,
    pub kind: SimpleBaselineKind,
    pub feature_schema: CategoricalFeatureSchema,
    /// Used by fixed-random and HDC encoders. Must be zero for one-hot RLS.
    pub encoded_dimension: usize,
    /// Used by fixed-random and HDC encoders. Must be zero for one-hot RLS.
    pub representation_seed: u64,
    pub rls: RlsConfig,
}

impl SimpleBaselineSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SIMPLE_BASELINE_SCHEMA_V1 {
            return Err(format!("unsupported simple-baseline schema: {}", self.schema));
        }
        self.feature_schema.validate()?;
        self.rls.validate()?;
        match self.kind {
            SimpleBaselineKind::OneHotRls => {
                if self.encoded_dimension != 0 || self.representation_seed != 0 {
                    return Err(
                        "one-hot RLS must zero irrelevant encoded-dimension/representation-seed fields"
                            .into(),
                    );
                }
            }
            SimpleBaselineKind::FixedRandomTanhRls | SimpleBaselineKind::VanillaHdcRls => {
                if self.encoded_dimension == 0 {
                    return Err("encoded dimension must be positive".into());
                }
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        canonical_hash(BASELINE_SPEC_HASH_DOMAIN, self)
    }
}

/// One frozen contract that emits all readout-matched B1 controls.
///
/// Using this type is preferred to hand-constructing three specs because it
/// prevents accidental changes in schema/RLS settings between conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedBaselineFamilySpec {
    pub feature_schema: CategoricalFeatureSchema,
    pub encoded_dimension: usize,
    pub representation_seed: u64,
    pub rls: RlsConfig,
}

impl MatchedBaselineFamilySpec {
    pub fn validate(&self) -> Result<(), String> {
        self.feature_schema.validate()?;
        self.rls.validate()?;
        if self.encoded_dimension == 0 {
            return Err("matched random/HDC dimension must be positive".into());
        }
        // Validate allocation feasibility before any baseline is constructed.
        let _ = OnlineRlsBinary::new(self.encoded_dimension, self.rls.clone())?;
        let one_hot_dimension = self.feature_schema.one_hot_dimension()?;
        let _ = OnlineRlsBinary::new(one_hot_dimension, self.rls.clone())?;
        Ok(())
    }

    pub fn specs(&self) -> Result<Vec<SimpleBaselineSpec>, String> {
        self.validate()?;
        Ok(vec![
            SimpleBaselineSpec {
                schema: SIMPLE_BASELINE_SCHEMA_V1.into(),
                kind: SimpleBaselineKind::OneHotRls,
                feature_schema: self.feature_schema.clone(),
                encoded_dimension: 0,
                representation_seed: 0,
                rls: self.rls.clone(),
            },
            SimpleBaselineSpec {
                schema: SIMPLE_BASELINE_SCHEMA_V1.into(),
                kind: SimpleBaselineKind::FixedRandomTanhRls,
                feature_schema: self.feature_schema.clone(),
                encoded_dimension: self.encoded_dimension,
                representation_seed: self.representation_seed,
                rls: self.rls.clone(),
            },
            SimpleBaselineSpec {
                schema: SIMPLE_BASELINE_SCHEMA_V1.into(),
                kind: SimpleBaselineKind::VanillaHdcRls,
                feature_schema: self.feature_schema.clone(),
                encoded_dimension: self.encoded_dimension,
                representation_seed: self.representation_seed,
                rls: self.rls.clone(),
            },
        ])
    }

    pub fn agents(&self) -> Result<Vec<SimpleBaselineAgent>, String> {
        self.specs()?
            .into_iter()
            .map(SimpleBaselineAgent::new)
            .collect()
    }
}

#[derive(Debug, Clone)]
struct FixedRandomTanhEncoder {
    schema: CategoricalFeatureSchema,
    input_dimension: usize,
    output_dimension: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl FixedRandomTanhEncoder {
    fn new(
        schema: CategoricalFeatureSchema,
        output_dimension: usize,
        seed: u64,
    ) -> Result<Self, String> {
        let input_dimension = schema.one_hot_dimension()?;
        if output_dimension == 0 {
            return Err("random-feature output dimension must be positive".into());
        }
        let weights_len = output_dimension
            .checked_mul(input_dimension)
            .ok_or_else(|| "random-feature matrix allocation overflow".to_string())?;
        let mut state = derived_seed(seed, RANDOM_FEATURE_DOMAIN, b"matrix-and-bias");
        let mut weights = Vec::with_capacity(weights_len);
        for _ in 0..weights_len {
            weights.push(uniform_signed(&mut state) as f32);
        }
        let mut bias = Vec::with_capacity(output_dimension);
        for _ in 0..output_dimension {
            bias.push(uniform_signed(&mut state) as f32);
        }
        Ok(Self {
            schema,
            input_dimension,
            output_dimension,
            weights,
            bias,
        })
    }

    fn encode(&self, assignment: &BTreeMap<String, i64>) -> Result<Vec<f64>, String> {
        let active = self.schema.active_one_hot_indices(assignment)?;
        let active_scale = 1.0 / (active.len() as f64).sqrt();
        let mut encoded = vec![0.0; self.output_dimension];
        for row in 0..self.output_dimension {
            let base = row * self.input_dimension;
            let mut activation = self.bias[row] as f64;
            for &column in &active {
                activation += self.weights[base + column] as f64 * active_scale;
            }
            encoded[row] = activation.tanh();
        }
        l2_normalize(&mut encoded)?;
        Ok(encoded)
    }

    fn state_bytes(&self) -> usize {
        (self.weights.len() + self.bias.len()) * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
struct VanillaHdcEncoder {
    schema: CategoricalFeatureSchema,
    dimension: usize,
    roles: BTreeMap<String, ContinuousHV>,
    values: BTreeMap<i64, ContinuousHV>,
}

impl VanillaHdcEncoder {
    fn new(
        schema: CategoricalFeatureSchema,
        dimension: usize,
        seed: u64,
    ) -> Result<Self, String> {
        schema.validate()?;
        if dimension == 0 {
            return Err("HDC output dimension must be positive".into());
        }
        let mut roles = BTreeMap::new();
        for feature in &schema.features {
            let role_seed = derived_seed(seed, HDC_ROLE_DOMAIN, feature.name.as_bytes());
            roles.insert(feature.name.clone(), ContinuousHV::random(dimension, role_seed));
        }
        let mut values = BTreeMap::new();
        for value in schema.unique_values()? {
            let value_seed = derived_seed(seed, HDC_VALUE_DOMAIN, &value.to_le_bytes());
            values.insert(value, ContinuousHV::random(dimension, value_seed));
        }
        Ok(Self {
            schema,
            dimension,
            roles,
            values,
        })
    }

    fn encode(&self, assignment: &BTreeMap<String, i64>) -> Result<Vec<f64>, String> {
        // Reuse the exact schema validation performed by one-hot encoding so the
        // HDC encoder cannot silently accept additional/missing learner features.
        let _ = self.schema.active_one_hot_indices(assignment)?;
        let mut bound = Vec::with_capacity(self.schema.features.len());
        for feature in &self.schema.features {
            let value = assignment
                .get(&feature.name)
                .copied()
                .ok_or_else(|| format!("assignment missing feature {}", feature.name))?;
            let role = self
                .roles
                .get(&feature.name)
                .ok_or_else(|| "HDC role table is incomplete".to_string())?;
            let value_hv = self
                .values
                .get(&value)
                .ok_or_else(|| format!("HDC value {value} is outside frozen schema"))?;
            bound.push(role.bind(value_hv));
        }
        let refs: Vec<&ContinuousHV> = bound.iter().collect();
        let bundled = ContinuousHV::bundle(&refs).normalize();
        if bundled.values.len() != self.dimension {
            return Err("HDC encoder produced unexpected dimension".into());
        }
        let mut encoded: Vec<f64> = bundled.values.iter().map(|value| *value as f64).collect();
        l2_normalize(&mut encoded)?;
        Ok(encoded)
    }

    fn state_bytes(&self) -> usize {
        let vector_count = self.roles.len() + self.values.len();
        vector_count * self.dimension * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
enum Encoder {
    OneHot(CategoricalFeatureSchema),
    FixedRandom(FixedRandomTanhEncoder),
    VanillaHdc(VanillaHdcEncoder),
}

impl Encoder {
    fn feature_dimension(&self) -> usize {
        match self {
            Self::OneHot(schema) => schema
                .one_hot_dimension()
                .expect("validated one-hot schema remains valid"),
            Self::FixedRandom(encoder) => encoder.output_dimension,
            Self::VanillaHdc(encoder) => encoder.dimension,
        }
    }

    fn encode(&self, assignment: &BTreeMap<String, i64>) -> Result<Vec<f64>, String> {
        match self {
            Self::OneHot(schema) => schema.normalized_one_hot(assignment),
            Self::FixedRandom(encoder) => encoder.encode(assignment),
            Self::VanillaHdc(encoder) => encoder.encode(assignment),
        }
    }

    fn fixed_state_bytes(&self) -> usize {
        match self {
            Self::OneHot(_) => 0,
            Self::FixedRandom(encoder) => encoder.state_bytes(),
            Self::VanillaHdc(encoder) => encoder.state_bytes(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineResourceFootprint {
    pub feature_dimension: usize,
    pub fixed_encoder_state_bytes: usize,
    pub readout_weight_bytes: usize,
    pub readout_covariance_bytes: usize,
    pub total_persistent_state_bytes: usize,
    pub trainable_parameters: usize,
    pub replay_examples: usize,
    pub temporal_state_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct SimpleBaselineAgent {
    spec: SimpleBaselineSpec,
    spec_digest: String,
    encoder: Encoder,
    readout: OnlineRlsBinary,
}

impl SimpleBaselineAgent {
    pub fn new(spec: SimpleBaselineSpec) -> Result<Self, String> {
        spec.validate()?;
        let spec_digest = spec.digest()?;
        let encoder = match spec.kind {
            SimpleBaselineKind::OneHotRls => Encoder::OneHot(spec.feature_schema.clone()),
            SimpleBaselineKind::FixedRandomTanhRls => Encoder::FixedRandom(
                FixedRandomTanhEncoder::new(
                    spec.feature_schema.clone(),
                    spec.encoded_dimension,
                    spec.representation_seed,
                )?,
            ),
            SimpleBaselineKind::VanillaHdcRls => Encoder::VanillaHdc(VanillaHdcEncoder::new(
                spec.feature_schema.clone(),
                spec.encoded_dimension,
                spec.representation_seed,
            )?),
        };
        let readout = OnlineRlsBinary::new(encoder.feature_dimension(), spec.rls.clone())?;
        Ok(Self {
            spec,
            spec_digest,
            encoder,
            readout,
        })
    }

    pub fn kind(&self) -> SimpleBaselineKind {
        self.spec.kind
    }

    pub fn spec(&self) -> &SimpleBaselineSpec {
        &self.spec
    }

    pub fn spec_digest(&self) -> &str {
        &self.spec_digest
    }

    pub fn encoded_features(
        &self,
        assignment: &BTreeMap<String, i64>,
    ) -> Result<Vec<f64>, String> {
        self.encoder.encode(assignment)
    }

    pub fn score(&self, assignment: &BTreeMap<String, i64>) -> Result<f64, String> {
        let encoded = self.encoder.encode(assignment)?;
        self.readout.score(&encoded)
    }

    pub fn predict(&self, assignment: &BTreeMap<String, i64>) -> Result<bool, String> {
        let encoded = self.encoder.encode(assignment)?;
        self.readout.predict(&encoded)
    }

    pub fn observe(
        &mut self,
        assignment: &BTreeMap<String, i64>,
        label: bool,
    ) -> Result<f64, String> {
        let encoded = self.encoder.encode(assignment)?;
        self.readout.update(&encoded, label)
    }

    pub fn updates(&self) -> usize {
        self.readout.updates()
    }

    pub fn resources(&self) -> Result<BaselineResourceFootprint, String> {
        let fixed_encoder_state_bytes = self.encoder.fixed_state_bytes();
        let readout_weight_bytes = self.readout.weight_bytes();
        let readout_covariance_bytes = self.readout.covariance_bytes();
        let total_persistent_state_bytes = fixed_encoder_state_bytes
            .checked_add(readout_weight_bytes)
            .and_then(|value| value.checked_add(readout_covariance_bytes))
            .ok_or_else(|| "baseline resource accounting overflow".to_string())?;
        Ok(BaselineResourceFootprint {
            feature_dimension: self.readout.input_dimension(),
            fixed_encoder_state_bytes,
            readout_weight_bytes,
            readout_covariance_bytes,
            total_persistent_state_bytes,
            trainable_parameters: self.readout.trainable_parameters(),
            replay_examples: 0,
            temporal_state_bytes: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> CategoricalFeatureSchema {
        CategoricalFeatureSchema {
            features: vec![
                CategoricalFeatureSpec {
                    name: "a".into(),
                    values: vec![0, 1],
                },
                CategoricalFeatureSpec {
                    name: "b".into(),
                    values: vec![0, 1],
                },
            ],
        }
    }

    fn assignment(a: i64, b: i64) -> BTreeMap<String, i64> {
        BTreeMap::from([("a".into(), a), ("b".into(), b)])
    }

    fn spec(kind: SimpleBaselineKind, dimension: usize, seed: u64) -> SimpleBaselineSpec {
        let (encoded_dimension, representation_seed) = match kind {
            SimpleBaselineKind::OneHotRls => (0, 0),
            _ => (dimension, seed),
        };
        SimpleBaselineSpec {
            schema: SIMPLE_BASELINE_SCHEMA_V1.into(),
            kind,
            feature_schema: schema(),
            encoded_dimension,
            representation_seed,
            rls: RlsConfig {
                ridge: 1.0,
                forgetting_factor: 1.0,
                include_bias: true,
            },
        }
    }

    fn family(dimension: usize, seed: u64) -> MatchedBaselineFamilySpec {
        MatchedBaselineFamilySpec {
            feature_schema: schema(),
            encoded_dimension: dimension,
            representation_seed: seed,
            rls: RlsConfig {
                ridge: 1.0,
                forgetting_factor: 1.0,
                include_bias: true,
            },
        }
    }

    #[test]
    fn schema_is_strict_and_one_hot_is_unit_norm() {
        let schema = schema();
        schema.validate().unwrap();
        assert_eq!(schema.one_hot_dimension().unwrap(), 4);
        let encoded = schema.normalized_one_hot(&assignment(1, 0)).unwrap();
        assert_eq!(encoded.iter().filter(|value| **value != 0.0).count(), 2);
        let norm: f64 = encoded.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);

        let mut invalid = assignment(1, 0);
        invalid.insert("hidden_task_id".into(), 7);
        assert!(schema.normalized_one_hot(&invalid).is_err());
    }

    #[test]
    fn rls_learns_a_simple_online_separator_without_replay() {
        let mut rls = OnlineRlsBinary::new(
            1,
            RlsConfig {
                ridge: 0.1,
                forgetting_factor: 1.0,
                include_bias: true,
            },
        )
        .unwrap();
        for _ in 0..12 {
            rls.update(&[-1.0], false).unwrap();
            rls.update(&[1.0], true).unwrap();
        }
        assert!(!rls.predict(&[-1.0]).unwrap());
        assert!(rls.predict(&[1.0]).unwrap());
        assert_eq!(rls.updates(), 24);
        assert!(rls.covariance_bytes() > rls.weight_bytes());
    }

    #[test]
    fn full_rls_rejects_multi_gigabyte_covariance_before_allocation() {
        let error = OnlineRlsBinary::new(16_384, RlsConfig::default()).unwrap_err();
        assert!(error.contains("hard ceiling"));
    }

    #[test]
    fn one_hot_spec_rejects_irrelevant_randomness_knobs() {
        let mut invalid = spec(SimpleBaselineKind::OneHotRls, 64, 7);
        invalid.representation_seed = 7;
        assert!(invalid.validate().is_err());
        invalid.representation_seed = 0;
        invalid.encoded_dimension = 64;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn fixed_random_encoder_is_seed_deterministic() {
        let first = SimpleBaselineAgent::new(spec(
            SimpleBaselineKind::FixedRandomTanhRls,
            64,
            17,
        ))
        .unwrap();
        let second = SimpleBaselineAgent::new(spec(
            SimpleBaselineKind::FixedRandomTanhRls,
            64,
            17,
        ))
        .unwrap();
        let different = SimpleBaselineAgent::new(spec(
            SimpleBaselineKind::FixedRandomTanhRls,
            64,
            18,
        ))
        .unwrap();
        let x = assignment(1, 0);
        let a = first.encoded_features(&x).unwrap();
        let b = second.encoded_features(&x).unwrap();
        let c = different.encoded_features(&x).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        let norm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn vanilla_hdc_encoder_is_seed_deterministic_and_normalized() {
        let first = SimpleBaselineAgent::new(spec(SimpleBaselineKind::VanillaHdcRls, 64, 23)).unwrap();
        let second = SimpleBaselineAgent::new(spec(SimpleBaselineKind::VanillaHdcRls, 64, 23)).unwrap();
        let different =
            SimpleBaselineAgent::new(spec(SimpleBaselineKind::VanillaHdcRls, 64, 24)).unwrap();
        let x = assignment(0, 1);
        let a = first.encoded_features(&x).unwrap();
        let b = second.encoded_features(&x).unwrap();
        let c = different.encoded_features(&x).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64);
        let norm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-8);
    }

    #[test]
    fn matched_family_freezes_one_readout_contract() {
        let family = family(128, 31);
        let specs = family.specs().unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].kind, SimpleBaselineKind::OneHotRls);
        assert_eq!(specs[1].kind, SimpleBaselineKind::FixedRandomTanhRls);
        assert_eq!(specs[2].kind, SimpleBaselineKind::VanillaHdcRls);
        assert_eq!(specs[0].rls, specs[1].rls);
        assert_eq!(specs[1].rls, specs[2].rls);
        assert_eq!(specs[0].feature_schema, specs[1].feature_schema);
        assert_eq!(specs[1].feature_schema, specs[2].feature_schema);
        assert_eq!(specs[0].encoded_dimension, 0);
        assert_eq!(specs[0].representation_seed, 0);
        assert_eq!(specs[1].encoded_dimension, 128);
        assert_eq!(specs[2].encoded_dimension, 128);
        assert_eq!(specs[1].representation_seed, 31);
        assert_eq!(specs[2].representation_seed, 31);
    }

    #[test]
    fn random_and_hdc_conditions_share_exact_readout_state_shape() {
        let agents = family(128, 31).agents().unwrap();
        let random = agents
            .iter()
            .find(|agent| agent.kind() == SimpleBaselineKind::FixedRandomTanhRls)
            .unwrap();
        let hdc = agents
            .iter()
            .find(|agent| agent.kind() == SimpleBaselineKind::VanillaHdcRls)
            .unwrap();
        let random_resources = random.resources().unwrap();
        let hdc_resources = hdc.resources().unwrap();
        assert_eq!(random_resources.feature_dimension, 128);
        assert_eq!(hdc_resources.feature_dimension, 128);
        assert_eq!(random_resources.readout_weight_bytes, hdc_resources.readout_weight_bytes);
        assert_eq!(
            random_resources.readout_covariance_bytes,
            hdc_resources.readout_covariance_bytes
        );
        assert_eq!(random_resources.trainable_parameters, hdc_resources.trainable_parameters);
        assert_eq!(random_resources.replay_examples, 0);
        assert_eq!(hdc_resources.replay_examples, 0);
        assert_eq!(random_resources.temporal_state_bytes, 0);
        assert_eq!(hdc_resources.temporal_state_bytes, 0);
    }

    #[test]
    fn encoder_state_is_label_independent() {
        let mut agent = SimpleBaselineAgent::new(spec(
            SimpleBaselineKind::FixedRandomTanhRls,
            32,
            41,
        ))
        .unwrap();
        let x = assignment(1, 1);
        let before = agent.encoded_features(&x).unwrap();
        for label in [true, false, true, true, false] {
            agent.observe(&x, label).unwrap();
        }
        let after = agent.encoded_features(&x).unwrap();
        assert_eq!(before, after);
        assert_eq!(agent.updates(), 5);
    }

    #[test]
    fn spec_digest_binds_kind_seed_schema_and_readout() {
        let raw = spec(SimpleBaselineKind::OneHotRls, 64, 5);
        let random = spec(SimpleBaselineKind::FixedRandomTanhRls, 64, 5);
        let other_seed = spec(SimpleBaselineKind::FixedRandomTanhRls, 64, 6);
        assert_ne!(raw.digest().unwrap(), random.digest().unwrap());
        assert_ne!(random.digest().unwrap(), other_seed.digest().unwrap());
    }

    #[test]
    fn resource_accounting_exposes_full_rls_covariance_cost() {
        let agent = SimpleBaselineAgent::new(spec(
            SimpleBaselineKind::FixedRandomTanhRls,
            64,
            7,
        ))
        .unwrap();
        let resources = agent.resources().unwrap();
        assert_eq!(resources.feature_dimension, 64);
        assert_eq!(resources.trainable_parameters, 65); // 64 features + bias
        assert_eq!(resources.readout_weight_bytes, 65 * 8);
        assert_eq!(resources.readout_covariance_bytes, 65 * 65 * 8);
        assert_eq!(resources.replay_examples, 0);
        assert_eq!(resources.temporal_state_bytes, 0);
        assert_eq!(
            resources.total_persistent_state_bytes,
            resources.fixed_encoder_state_bytes
                + resources.readout_weight_bytes
                + resources.readout_covariance_bytes
        );
    }
}