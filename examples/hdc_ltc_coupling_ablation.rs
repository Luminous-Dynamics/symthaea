// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Mechanism Integrity and History-Sensitivity Harness (formerly "HDC-LTC coupling
//! ablation" — renamed 2026-07-28, plan §14, to match its earned scope after two disclosed task
//! flaws; see below).
//!
//! Design doc: `SYMTHAEA_HDC_LTC_COUPLING_ABLATION_PLAN.md` (repo root). Read that file before
//! interpreting output — this harness implements its §3 arm set, §5 mechanical
//! intervention-integrity contract, §6 schedule matrix, and §9 reporting rules.
//!
//! **2026-07-28 status (plan §14): FROZEN as a comparative-capability benchmark, DEMOTED to a
//! regression/integrity test.** Original research question — does `HdcLtcBridge`'s learned
//! closed-form LTC temporal coupling produce measurable capability beyond simpler/conventional
//! temporal-state mechanisms? — is **not answered** by this harness: `PE`/`novelty_discrimination`
//! were found to not test predictive capability at all (plan §12-13, two independent flaws), and
//! this harness's fixed-content corpus cannot be repaired into a valid comparative benchmark
//! in-place (per plan §14, that needs an information-theoretically-validated, state-aliasing
//! corpus — a new, separately-designed and preregistered task, not a patch to this file). What
//! remains earned and still trustworthy here: confirming each arm's declared mechanism actually
//! runs (§5's mechanical intervention-integrity contract), confirming `Static` carries zero
//! history (validated: reads exactly 0.0000 on `order_sensitivity` in every run to date),
//! detecting whether a representation changes under history/order swaps (`order_sensitivity` —
//! proves order-DEPENDENCE, not forecasting; see its own doc below), rough compute-cost
//! accounting, and regression-testing the call-count bookkeeping itself. Keep using this file for
//! those purposes; do not use it to argue one mechanism predicts better than another.
//!
//! **2026-07-28 PE-metric fix (plan §12 → §13)**: `run_developmental`'s training/eval target used
//! to be the same item's own encoding (reconstruction), which structurally favored
//! less-historically-entangled arms for reasons unrelated to predictive capability. Fixed to
//! genuine next-item prediction — see that function's doc for detail. `PE(A)`/`PE(B)`/
//! `novelty_discrimination` numbers from before this fix (plan §11) are not comparable to numbers
//! produced by the current code.
//!
//! # Scope decisions not literally spelled out in the plan (disclosed here, not silently assumed)
//! - **Shared encoder**: the plan requires all arms share "encoder, dimensionality, projection
//!   boundary, readout, ... initialization, seeds." Symthaea's real perceptual encoder is a
//!   whole-cognitive-loop concern; this experiment isolates the temporal-STATE mechanism, not
//!   perceptual encoding quality, so a simple deterministic hashing-trick trigram-bag encoder
//!   (`encode_text`) stands in for it, applied bit-identically to every arm.
//! - **Shared projection boundary**: `init_projection`/`project_to_hdc`/`project_from_hdc` below
//!   are copied verbatim (same PRNG, same seed offsets +100000/+200000) from
//!   `src/hdc_ltc_bridge.rs`'s private methods, so the HDC–LTC arm's internal projections and the
//!   three baseline arms' projections are bit-identical when constructed from the same seed.
//! - **Shared readout + optimizer**: the three baseline arms train a linear `output_projection`
//!   with the exact gradient formula `HdcLtcBridge::update_projections` uses
//!   (`apply_readout_gradient` below) — same normalize-by-state-norm scheme, same learning rate.
//!   The HDC–LTC arm's readout is the bridge's own (identical formula, verified by construction
//!   since it's the same code).
//! - **Regime separation**: Keystone's CL(body) − CL(coda) has no analog below the consciousness
//!   layer. Substituted with `state_diversity(varied) − state_diversity(repetitive-coda)`, using
//!   the identical variance→sigmoid formula `HdcLtcBridge::update_diversity` uses, computed
//!   identically for every arm's state vector.
//! - **Retention/recall** (plan §7): not implemented in this bounded pilot — disclosed, not
//!   silently dropped.
//! - **Schedule matrix** (plan §6): operationalized with two content streams A/B — "blocked" =
//!   all of A's reps then all of B's; "interleaved" = A/B alternated token-by-token; "irregular"
//!   = interleaved with a variable, deterministic `dt` sequence instead of constant `dt`. EMA's
//!   alpha update ignores `dt` by design (plan §3 says "fixed exponential-decay") — this makes it
//!   deliberately timing-blind, itself a point of comparison against HDC–LTC's time-parameterized
//!   evolution, not an oversight.
//!
//! Dims here (`input_dim=32`, `hdc_dim=512`) are deliberately small for a **bounded pilot** per
//! plan §8 — a confirmatory run with fresh seeds should reconsider scale before any claim is
//! finalized. Run: `cargo run --release --example hdc_ltc_coupling_ablation`

use std::time::Instant;
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════
// SHARED PROJECTION / ENCODER (bit-identical across all arms)
// ═══════════════════════════════════════════════════════════════════════════

/// Copied verbatim from `HdcLtcBridge::init_projection` so every arm's projection matrices are
/// bit-identical when given the same seed. Do not "clean up" the PRNG here without also updating
/// the bridge — divergence would silently break the shared-projection guarantee this experiment
/// depends on.
fn init_projection(input_dim: usize, output_dim: usize, seed: u64) -> Vec<f32> {
    let mut projection = Vec::with_capacity(input_dim * output_dim);
    let mut state = seed;
    let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
    for _ in 0..(input_dim * output_dim) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        projection.push(normalized * scale);
    }
    projection
}

/// Copied from `HdcLtcBridge::project_to_hdc` (tanh-bounded row accumulation).
fn project_to_hdc(input: &[f32], proj: &[f32], input_dim: usize, hdc_dim: usize) -> ContinuousHV {
    let mut values = vec![0.0f32; hdc_dim];
    for i in 0..input_dim.min(input.len()) {
        let x = input[i];
        if x.abs() < 1e-10 {
            continue;
        }
        let row = &proj[i * hdc_dim..(i + 1) * hdc_dim];
        for (v, &w) in values.iter_mut().zip(row.iter()) {
            *v += x * w;
        }
    }
    for v in values.iter_mut() {
        *v = v.tanh();
    }
    ContinuousHV::from_values(values)
}

/// Copied from `HdcLtcBridge::project_from_hdc` (row accumulation + norm restoration).
fn project_from_hdc(
    hv: &ContinuousHV,
    proj: &[f32],
    hdc_dim: usize,
    output_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; output_dim];
    for j in 0..hdc_dim {
        let x = hv.values[j];
        let row = &proj[j * output_dim..(j + 1) * output_dim];
        for (o, &w) in output.iter_mut().zip(row.iter()) {
            *o += x * w;
        }
    }
    let state_norm = hv.norm();
    if state_norm > 1e-30 {
        for o in output.iter_mut() {
            *o /= state_norm;
        }
    }
    output
}

/// Copied from `HdcLtcBridge::update_projections`'s gradient formula, generalized to any
/// arm's `(hdc_state, output_projection)` pair. Same normalize-by-state-norm scheme as the
/// forward pass, so every arm's readout trains under an identical optimizer.
fn apply_readout_gradient(
    output_projection: &mut [f32],
    hdc_state: &ContinuousHV,
    target: &[f32],
    output: &[f32],
    hdc_dim: usize,
    output_dim: usize,
    learning_rate: f32,
) {
    let state_norm = hdc_state.norm();
    if state_norm <= 1e-30 {
        return;
    }
    let inv_norm = 1.0 / state_norm;
    let errors: Vec<f32> = output
        .iter()
        .zip(target.iter())
        .map(|(o, t)| o - t)
        .collect();
    for i in 0..output_dim {
        for j in 0..hdc_dim {
            let grad = errors[i] * hdc_state.values[j] * inv_norm;
            output_projection[j * output_dim + i] -= learning_rate * grad;
        }
    }
}

/// Shared deterministic hashing-trick trigram-bag encoder — see module doc's "Shared encoder"
/// scope note. Same text always produces the same vector; similar texts (shared trigrams)
/// produce correlated vectors, which is all the task battery below needs.
fn encode_text(text: &str, input_dim: usize) -> Vec<f32> {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut buckets = vec![0.0f32; input_dim];
    if chars.len() < 3 {
        buckets[0] = 1.0;
        return buckets;
    }
    for w in chars.windows(3) {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for c in w {
            hash ^= *c as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        buckets[(hash as usize) % input_dim] += 1.0;
    }
    let norm = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for b in buckets.iter_mut() {
            *b /= norm;
        }
    }
    buckets
}

// ═══════════════════════════════════════════════════════════════════════════
// §5 MECHANICAL INTERVENTION-INTEGRITY CONTRACT
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
enum TemporalStateMode {
    HdcLtc,
    Static,
    Ema { alpha: f32 },
    PermutationVsa,
}

impl std::fmt::Display for TemporalStateMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalStateMode::HdcLtc => write!(f, "HdcLtc"),
            TemporalStateMode::Static => write!(f, "Static"),
            TemporalStateMode::Ema { alpha } => write!(f, "Ema(a={alpha})"),
            TemporalStateMode::PermutationVsa => write!(f, "PermutationVsa"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CallCounts {
    hdc_ltc_predict: u64,
    hdc_ltc_train: u64,
    ema_updates: u64,
    permutation_ops: u64,
}

/// Hard-failure check from plan §5's required-assertions table. Aborts the process (not a
/// warning) on mismatch — a run whose declared arm doesn't match its measured call pattern must
/// not silently complete and produce numbers, per the `no_engine` naming-failure lesson this
/// whole plan exists to correct.
fn assert_mechanical_integrity(mode: TemporalStateMode, counts: CallCounts, trained: bool) {
    match mode {
        TemporalStateMode::Static => {
            assert_eq!(
                counts.hdc_ltc_predict, 0,
                "Static: HDC-LTC predict must be 0"
            );
            assert_eq!(counts.hdc_ltc_train, 0, "Static: HDC-LTC train must be 0");
            assert_eq!(counts.ema_updates, 0, "Static: EMA updates must be 0");
            assert_eq!(
                counts.permutation_ops, 0,
                "Static: permutation ops must be 0"
            );
        }
        TemporalStateMode::Ema { .. } => {
            assert_eq!(counts.hdc_ltc_predict, 0, "EMA: HDC-LTC predict must be 0");
            assert_eq!(counts.hdc_ltc_train, 0, "EMA: HDC-LTC train must be 0");
            assert_eq!(counts.permutation_ops, 0, "EMA: permutation ops must be 0");
            assert!(counts.ema_updates > 0, "EMA: EMA updates must be > 0");
        }
        TemporalStateMode::PermutationVsa => {
            assert_eq!(
                counts.hdc_ltc_predict, 0,
                "Permutation-VSA: HDC-LTC predict must be 0"
            );
            assert_eq!(
                counts.hdc_ltc_train, 0,
                "Permutation-VSA: HDC-LTC train must be 0"
            );
            assert_eq!(
                counts.ema_updates, 0,
                "Permutation-VSA: EMA updates must be 0"
            );
            assert!(
                counts.permutation_ops > 0,
                "Permutation-VSA: permutation/bind ops must be > 0"
            );
        }
        TemporalStateMode::HdcLtc => {
            assert!(
                counts.hdc_ltc_predict > 0,
                "HDC-LTC: predict calls must be > 0"
            );
            if trained {
                assert!(
                    counts.hdc_ltc_train > 0,
                    "HDC-LTC: train calls must be > 0 during training"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ARM TRAIT + IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

trait Arm {
    fn mode(&self) -> TemporalStateMode;
    /// Advance state on this input (no training).
    fn step(&mut self, input: &[f32], dt: f32);
    /// Train the readout (and, for HdcLtc, the recurrent weights) on (input, target); returns
    /// MSE loss. Also advances state exactly as `step` would for non-HdcLtc arms, matching
    /// HdcLtcBridge's own train_step semantics (pure w.r.t. live evolution state).
    fn train_step(&mut self, input: &[f32], target: &[f32], dt: f32, lr: f32) -> f32;
    /// Pure prediction from current state at the given horizon; must not observably mutate state.
    fn predict(&mut self, input: &[f32], horizon: f32) -> Vec<f32>;
    fn state_diversity(&self) -> f32;
    fn call_counts(&self) -> CallCounts;
}

fn diversity_from_values(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for &v in values {
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    1.0 / (1.0 + (-variance.sqrt() * 10.0).exp())
}

// --- HdcLtc arm: thin wrapper around the real production bridge --------------------------------

struct HdcLtcArm {
    bridge: symthaea::hdc_ltc_bridge::HdcLtcBridge,
    output_dim: usize,
    counts: CallCounts,
}

impl HdcLtcArm {
    fn new(input_dim: usize, output_dim: usize, hdc_dim: usize, seed: u64) -> Self {
        let config = symthaea::hdc_ltc_bridge::HdcLtcBridgeConfig {
            input_dim,
            output_dim,
            layer_sizes: vec![2, 2],
            hdc_dim,
            seed,
            ..Default::default()
        };
        Self {
            bridge: symthaea::hdc_ltc_bridge::HdcLtcBridge::new(config),
            output_dim,
            counts: CallCounts::default(),
        }
    }
}

impl Arm for HdcLtcArm {
    fn mode(&self) -> TemporalStateMode {
        TemporalStateMode::HdcLtc
    }
    fn step(&mut self, input: &[f32], dt: f32) {
        let arr = ndarray::Array1::from_vec(input.to_vec());
        let _ = self.bridge.step(&arr, dt);
    }
    fn train_step(&mut self, input: &[f32], target: &[f32], dt: f32, lr: f32) -> f32 {
        let in_arr = ndarray::Array1::from_vec(input.to_vec());
        let tgt_arr = ndarray::Array1::from_vec(target.to_vec());
        let loss = self
            .bridge
            .train_step(&in_arr, &tgt_arr, dt, lr)
            .unwrap_or(f32::NAN);
        self.counts.hdc_ltc_train += 1;
        // train_step is pure w.r.t. evolution state (see bridge doc) — advance state for real
        // via a real step, matching what the baseline arms' train_step does.
        self.step(input, dt);
        loss
    }
    fn predict(&mut self, input: &[f32], horizon: f32) -> Vec<f32> {
        let arr = ndarray::Array1::from_vec(input.to_vec());
        self.counts.hdc_ltc_predict += 1;
        self.bridge
            .predict_forward(&arr, horizon)
            .map(|a| a.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.output_dim])
    }
    fn state_diversity(&self) -> f32 {
        self.bridge.state_diversity()
    }
    fn call_counts(&self) -> CallCounts {
        self.counts
    }
}

// --- Static arm: no state carry across steps ----------------------------------------------------

struct StaticArm {
    input_dim: usize,
    hdc_dim: usize,
    output_dim: usize,
    input_projection: Vec<f32>,
    output_projection: Vec<f32>,
    last_state: ContinuousHV,
    counts: CallCounts,
}

impl StaticArm {
    fn new(input_dim: usize, output_dim: usize, hdc_dim: usize, seed: u64) -> Self {
        Self {
            input_dim,
            hdc_dim,
            output_dim,
            input_projection: init_projection(input_dim, hdc_dim, seed + 100_000),
            output_projection: init_projection(hdc_dim, output_dim, seed + 200_000),
            last_state: ContinuousHV::zero(hdc_dim),
            counts: CallCounts::default(),
        }
    }
}

impl Arm for StaticArm {
    fn mode(&self) -> TemporalStateMode {
        TemporalStateMode::Static
    }
    fn step(&mut self, input: &[f32], _dt: f32) {
        // No state carry: state is entirely overwritten by the current input's projection.
        self.last_state =
            project_to_hdc(input, &self.input_projection, self.input_dim, self.hdc_dim);
    }
    fn train_step(&mut self, input: &[f32], target: &[f32], dt: f32, lr: f32) -> f32 {
        self.step(input, dt);
        let output = project_from_hdc(
            &self.last_state,
            &self.output_projection,
            self.hdc_dim,
            self.output_dim,
        );
        let loss = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / target.len() as f32;
        apply_readout_gradient(
            &mut self.output_projection,
            &self.last_state,
            target,
            &output,
            self.hdc_dim,
            self.output_dim,
            lr,
        );
        loss
    }
    fn predict(&mut self, input: &[f32], _horizon: f32) -> Vec<f32> {
        // Pure: predicting from a hypothetical next input's own (stateless) projection.
        let hv = project_to_hdc(input, &self.input_projection, self.input_dim, self.hdc_dim);
        project_from_hdc(&hv, &self.output_projection, self.hdc_dim, self.output_dim)
    }
    fn state_diversity(&self) -> f32 {
        diversity_from_values(&self.last_state.values)
    }
    fn call_counts(&self) -> CallCounts {
        self.counts
    }
}

// --- EMA arm: fixed exponential-decay state update, timing-blind by design (plan §3) ------------

struct EmaArm {
    input_dim: usize,
    hdc_dim: usize,
    output_dim: usize,
    alpha: f32,
    input_projection: Vec<f32>,
    output_projection: Vec<f32>,
    state: ContinuousHV,
    counts: CallCounts,
}

impl EmaArm {
    fn new(input_dim: usize, output_dim: usize, hdc_dim: usize, alpha: f32, seed: u64) -> Self {
        Self {
            input_dim,
            hdc_dim,
            output_dim,
            alpha,
            input_projection: init_projection(input_dim, hdc_dim, seed + 100_000),
            output_projection: init_projection(hdc_dim, output_dim, seed + 200_000),
            state: ContinuousHV::zero(hdc_dim),
            counts: CallCounts::default(),
        }
    }
}

impl Arm for EmaArm {
    fn mode(&self) -> TemporalStateMode {
        TemporalStateMode::Ema { alpha: self.alpha }
    }
    fn step(&mut self, input: &[f32], _dt: f32) {
        let x = project_to_hdc(input, &self.input_projection, self.input_dim, self.hdc_dim);
        for (s, xv) in self.state.values.iter_mut().zip(x.values.iter()) {
            *s = self.alpha * xv + (1.0 - self.alpha) * *s;
        }
        self.counts.ema_updates += 1;
    }
    fn train_step(&mut self, input: &[f32], target: &[f32], dt: f32, lr: f32) -> f32 {
        self.step(input, dt);
        let output = project_from_hdc(
            &self.state,
            &self.output_projection,
            self.hdc_dim,
            self.output_dim,
        );
        let loss = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / target.len() as f32;
        apply_readout_gradient(
            &mut self.output_projection,
            &self.state,
            target,
            &output,
            self.hdc_dim,
            self.output_dim,
            lr,
        );
        loss
    }
    fn predict(&mut self, input: &[f32], _horizon: f32) -> Vec<f32> {
        // Pure: evolve a scratch copy of the EMA state, never touch self.state.
        let x = project_to_hdc(input, &self.input_projection, self.input_dim, self.hdc_dim);
        let mut scratch = self.state.clone();
        for (s, xv) in scratch.values.iter_mut().zip(x.values.iter()) {
            *s = self.alpha * xv + (1.0 - self.alpha) * *s;
        }
        project_from_hdc(
            &scratch,
            &self.output_projection,
            self.hdc_dim,
            self.output_dim,
        )
    }
    fn state_diversity(&self) -> f32 {
        diversity_from_values(&self.state.values)
    }
    fn call_counts(&self) -> CallCounts {
        self.counts
    }
}

// --- Permutation-VSA arm: classic discrete sequence trace (Kanerva-style) -----------------------

struct PermutationArm {
    input_dim: usize,
    hdc_dim: usize,
    output_dim: usize,
    input_projection: Vec<f32>,
    output_projection: Vec<f32>,
    state: ContinuousHV,
    counts: CallCounts,
}

impl PermutationArm {
    fn new(input_dim: usize, output_dim: usize, hdc_dim: usize, seed: u64) -> Self {
        Self {
            input_dim,
            hdc_dim,
            output_dim,
            input_projection: init_projection(input_dim, hdc_dim, seed + 100_000),
            output_projection: init_projection(hdc_dim, output_dim, seed + 200_000),
            state: ContinuousHV::zero(hdc_dim),
            counts: CallCounts::default(),
        }
    }
}

impl Arm for PermutationArm {
    fn mode(&self) -> TemporalStateMode {
        TemporalStateMode::PermutationVsa
    }
    fn step(&mut self, input: &[f32], _dt: f32) {
        // state_t = bundle(permute(state_{t-1}, 1), x_t): a decaying positional trace of the
        // whole history — permutation depth encodes recency/order, bundling superposes.
        let x = project_to_hdc(input, &self.input_projection, self.input_dim, self.hdc_dim);
        let permuted = self.state.permute(1);
        self.state = ContinuousHV::bundle(&[&permuted, &x]);
        self.counts.permutation_ops += 1;
    }
    fn train_step(&mut self, input: &[f32], target: &[f32], dt: f32, lr: f32) -> f32 {
        self.step(input, dt);
        let output = project_from_hdc(
            &self.state,
            &self.output_projection,
            self.hdc_dim,
            self.output_dim,
        );
        let loss = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / target.len() as f32;
        apply_readout_gradient(
            &mut self.output_projection,
            &self.state,
            target,
            &output,
            self.hdc_dim,
            self.output_dim,
            lr,
        );
        loss
    }
    fn predict(&mut self, input: &[f32], _horizon: f32) -> Vec<f32> {
        let x = project_to_hdc(input, &self.input_projection, self.input_dim, self.hdc_dim);
        let permuted = self.state.permute(1);
        let scratch = ContinuousHV::bundle(&[&permuted, &x]);
        project_from_hdc(
            &scratch,
            &self.output_projection,
            self.hdc_dim,
            self.output_dim,
        )
    }
    fn state_diversity(&self) -> f32 {
        diversity_from_values(&self.state.values)
    }
    fn call_counts(&self) -> CallCounts {
        self.counts
    }
}

fn make_arm(
    mode: TemporalStateMode,
    input_dim: usize,
    output_dim: usize,
    hdc_dim: usize,
    seed: u64,
) -> Box<dyn Arm> {
    match mode {
        TemporalStateMode::HdcLtc => Box::new(HdcLtcArm::new(input_dim, output_dim, hdc_dim, seed)),
        TemporalStateMode::Static => Box::new(StaticArm::new(input_dim, output_dim, hdc_dim, seed)),
        TemporalStateMode::Ema { alpha } => {
            Box::new(EmaArm::new(input_dim, output_dim, hdc_dim, alpha, seed))
        }
        TemporalStateMode::PermutationVsa => {
            Box::new(PermutationArm::new(input_dim, output_dim, hdc_dim, seed))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTENT (small pilot corpora — two streams for blocked/interleaved schedules)
// ═══════════════════════════════════════════════════════════════════════════

fn stream_a() -> Vec<&'static str> {
    vec![
        "the water cycle moves moisture from oceans to clouds",
        "gratitude settles quietly over a calm morning",
        "the reactor coolant temperature is rising",
        "two plus two equals four and four plus four is eight",
    ]
}

fn stream_b() -> Vec<&'static str> {
    vec![
        "warning unauthorized access detected on the mesh network",
        "the old oak tree stood in that field for years",
        "the market fell on news of the supply shortage",
        "complete the safety checklist before enabling the motor",
    ]
}

fn novel_stream() -> Vec<&'static str> {
    vec![
        "volcanic ash grounded flights across the corridor",
        "he whittled a small boat from driftwood",
        "neutron flux in channel seven exceeds threshold",
        "seven times eight is fifty six",
        "the choir final chord hung in the rafters",
        "checksum mismatch detected in the boot partition",
    ]
}

fn coda_input() -> &'static str {
    "the system hums quietly in the background"
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Schedule {
    Blocked,
    Interleaved,
    Irregular,
}

impl std::fmt::Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Schedule::Blocked => write!(f, "blocked"),
            Schedule::Interleaved => write!(f, "interleaved"),
            Schedule::Irregular => write!(f, "irregular"),
        }
    }
}

const DT_BASE: f32 = 0.02;
/// Deterministic irregular-gap pattern (plan §6): cycles through varied simulated elapsed time
/// instead of a constant dt.
const DT_IRREGULAR: &[f32] = &[0.02, 0.05, 0.10, 0.02, 0.08];

fn dt_for(schedule: Schedule, step_idx: usize) -> f32 {
    match schedule {
        Schedule::Irregular => DT_IRREGULAR[step_idx % DT_IRREGULAR.len()],
        _ => DT_BASE,
    }
}

/// Build the ordered (content, is_stream_a) presentation sequence for one repetition round under
/// the given schedule. Blocked: all of A then all of B. Interleaved/Irregular: A0,B0,A1,B1,...
fn presentation_order(schedule: Schedule) -> Vec<(&'static str, bool)> {
    let a = stream_a();
    let b = stream_b();
    match schedule {
        Schedule::Blocked => a
            .into_iter()
            .map(|s| (s, true))
            .chain(b.into_iter().map(|s| (s, false)))
            .collect(),
        Schedule::Interleaved | Schedule::Irregular => a
            .into_iter()
            .zip(b)
            .flat_map(|(x, y)| [(x, true), (y, false)])
            .collect(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════════════════════════════════════

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

struct RunResult {
    mode: TemporalStateMode,
    schedule: Schedule,
    pe_a: f64,
    pe_b: f64,
    pe_novel: f64,
    novelty_discrimination: f64,
    order_sensitivity: f64,
    regime_separation: f64,
    mean_step_us: f64,
    counts: CallCounts,
}

const INPUT_DIM: usize = 32;
const HDC_DIM: usize = 512;
const DEFAULT_REPS: usize = 12;
const LR: f32 = 0.02;

/// Developmental run (plan §4 primary): train `mode` from scratch under `schedule`, then measure
/// the pre-declared task battery (plan §7, minus retention/recall — see module doc). `reps` is
/// runtime-configurable (unlike dims) specifically so a confirmatory pass can cheaply re-test
/// whether a pilot-scale null (e.g. `order_sensitivity`) survives more training reps, without a
/// recompile per scale.
///
/// # Terminology correction (2026-07-28, plan §14): "sensitivity," not "anticipation"
/// `order_sensitivity` (below) proves the state's representation depends on preceding order/
/// history — it does NOT prove genuine forecasting (predicting distinct correct futures FROM
/// that history). This module used to call it "order-anticipation"; comments below are corrected
/// to "order/history-swap sensitivity." Earning the stronger "anticipation" claim needs a task
/// where the same current item requires different correct next-predictions depending on earlier
/// context — not attempted by this harness (see plan §14).
///
/// # PE-metric fix (2026-07-28, plan §12 → §13)
/// The training/eval target used to be the SAME item's own encoding (a reconstruction task),
/// which structurally favors less-historically-entangled states (Static most) for reasons
/// unrelated to predictive capability — see §12's found flaw. Fixed here: the target is now the
/// NEXT item in the presentation sequence, a genuine next-item-prediction task. The
/// order/history-swap sensitivity probe is unaffected by this fix and deliberately left as
/// same-item reconstruction-conditioned-on-history — it's a within-arm swapped-vs-clean
/// DIFFERENCE, so the absolute reconstruction-ease confound cancels out (already validated: Static reads exactly
/// 0.0000, the correct answer for a zero-history arm regardless of which target convention is
/// used).
fn run_developmental(
    mode: TemporalStateMode,
    schedule: Schedule,
    seed: u64,
    reps: usize,
) -> RunResult {
    let mut arm = make_arm(mode, INPUT_DIM, INPUT_DIM, HDC_DIM, seed);
    // Declared-vs-actual identity check (plan §5's own theme, applied to the arm itself, not
    // just its call counts): the constructed arm must self-report the mode it was asked for.
    assert_eq!(
        arm.mode(),
        mode,
        "constructed arm's own mode() disagrees with the mode it was requested with"
    );
    let order = presentation_order(schedule);
    let mut step_us: Vec<f64> = Vec::new();
    let mut global_step = 0usize;

    // Train: genuine next-item prediction. Flatten `reps` repetitions of `order` into one
    // sequence so "next item" wraps naturally across repetition boundaries too (the last item of
    // rep k predicts the first item of rep k+1) — not just within a single repetition.
    let flat_texts: Vec<&str> = (0..reps)
        .flat_map(|_| order.iter().map(|&(t, _)| t))
        .collect();
    for i in 0..flat_texts.len().saturating_sub(1) {
        let x = encode_text(flat_texts[i], INPUT_DIM);
        let next = encode_text(flat_texts[i + 1], INPUT_DIM);
        let dt = dt_for(schedule, global_step);
        let t = Instant::now();
        let _ = arm.train_step(&x, &next, dt, LR);
        step_us.push(t.elapsed().as_micros() as f64);
        global_step += 1;
    }

    // Task: next-item prediction error on A and B (post-training, held-out pass — no further
    // training). windows(2) so the target is genuinely the FOLLOWING item, matching the training
    // objective above — an eval/train target mismatch would itself be a new flaw.
    let measure_pe = |arm: &mut Box<dyn Arm>, texts: &[&str]| -> f64 {
        let mut pes = Vec::new();
        for w in texts.windows(2) {
            let x = encode_text(w[0], INPUT_DIM);
            let next_x = encode_text(w[1], INPUT_DIM);
            let dt = DT_BASE;
            let pred = arm.predict(&x, dt);
            let pe = pred
                .iter()
                .zip(next_x.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / next_x.len() as f32;
            pes.push(pe as f64);
            arm.step(&x, dt);
        }
        mean(&pes)
    };
    let pe_a = measure_pe(&mut arm, &stream_a());
    let pe_b = measure_pe(&mut arm, &stream_b());
    let pe_novel = measure_pe(&mut arm, &novel_stream());
    let novelty_discrimination = pe_novel - (pe_a + pe_b) / 2.0;

    // Task: order/history-swap sensitivity probe within stream A (same idea as keystone_ab.rs).
    let a = stream_a();
    let n = a.len();
    let mut swapped_pes: Vec<f64> = Vec::new();
    let mut clean_pes_at: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut probe_positions: Vec<(usize, usize)> = Vec::new();
    for r in 0..8usize {
        let (p, q) = (r % n, (r + 2) % n);
        let swapped_rep = r % 2 == 1;
        if swapped_rep && p != q {
            probe_positions.push((p, q));
        }
        for pos in 0..n {
            let content = if swapped_rep && pos == p && p != q {
                a[q]
            } else if swapped_rep && pos == q && p != q {
                a[p]
            } else {
                a[pos]
            };
            let x = encode_text(content, INPUT_DIM);
            let pred = arm.predict(&x, DT_BASE);
            let pe = (pred
                .iter()
                .zip(x.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / x.len() as f32) as f64;
            arm.step(&x, DT_BASE);
            if swapped_rep && (pos == p || pos == q) && p != q {
                swapped_pes.push(pe);
            } else if !swapped_rep {
                clean_pes_at[pos].push(pe);
            }
        }
    }
    let control_pes: Vec<f64> = probe_positions
        .iter()
        .flat_map(|&(p, q)| {
            clean_pes_at[p]
                .iter()
                .chain(clean_pes_at[q].iter())
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let order_sensitivity = mean(&swapped_pes) - mean(&control_pes);

    // Task: regime separation — state_diversity(varied A+B pass) vs state_diversity(repetitive
    // coda). See module doc's "Regime separation" scope note.
    let mut varied_divs: Vec<f64> = Vec::new();
    for &(text, _) in order.iter() {
        let x = encode_text(text, INPUT_DIM);
        arm.step(&x, DT_BASE);
        varied_divs.push(arm.state_diversity() as f64);
    }
    let mut coda_divs: Vec<f64> = Vec::new();
    let coda_x = encode_text(coda_input(), INPUT_DIM);
    for _ in 0..12 {
        arm.step(&coda_x, DT_BASE);
        coda_divs.push(arm.state_diversity() as f64);
    }
    let regime_separation = mean(&varied_divs) - mean(&coda_divs);

    let counts = arm.call_counts();
    // §5: hard-fail if the declared arm's measured call pattern doesn't match its name.
    assert_mechanical_integrity(mode, counts, true);

    RunResult {
        mode,
        schedule,
        pe_a,
        pe_b,
        pe_novel,
        novelty_discrimination,
        order_sensitivity,
        regime_separation,
        mean_step_us: mean(&step_us),
        counts,
    }
}

/// CLI usage: `hdc_ltc_coupling_ablation [reps] [seed1,seed2,...]`
/// Defaults reproduce the exact 2026-07-28 bounded pilot (§11): reps=12, seeds=1001,1002,1003.
/// A confirmatory pass (plan §8) must pass a fresh, disjoint seed list; re-testing whether the
/// pilot's order/history-swap sensitivity null was a scale artifact means also raising reps.
fn parse_args() -> (usize, Vec<u64>) {
    let args: Vec<String> = std::env::args().collect();
    let reps = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPS);
    let seeds = args
        .get(2)
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.parse::<u64>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![1001, 1002, 1003]);
    (reps, seeds)
}

fn main() {
    let (reps, pilot_seeds) = parse_args();
    let is_default_scale = reps == DEFAULT_REPS && pilot_seeds.as_slice() == [1001u64, 1002, 1003];

    println!("=== HDC-LTC COUPLING ABLATION ===");
    println!("design doc: SYMTHAEA_HDC_LTC_COUPLING_ABLATION_PLAN.md");
    println!(
        "dims: input={INPUT_DIM} hdc={HDC_DIM} | reps={reps} lr={LR} | arms=4 schedules=3 | seeds={pilot_seeds:?}{}\n",
        if is_default_scale {
            " (same reps/seeds as the original §11 pilot — but §12 fixed the PE-metric's \
             target to genuine next-item prediction, so these numbers are NOT directly \
             comparable to §11's; see §13)"
        } else {
            " (NOT the original pilot's reps/seeds — treat as a separate run, e.g. confirmatory §8)"
        }
    );

    let modes = [
        TemporalStateMode::HdcLtc,
        TemporalStateMode::Static,
        TemporalStateMode::Ema { alpha: 0.3 },
        TemporalStateMode::PermutationVsa,
    ];
    let schedules = [
        Schedule::Blocked,
        Schedule::Interleaved,
        Schedule::Irregular,
    ];
    let pilot_seeds: &[u64] = &pilot_seeds;

    let mut results: Vec<RunResult> = Vec::new();
    for &schedule in &schedules {
        for &mode in &modes {
            for &seed in pilot_seeds {
                let r = run_developmental(mode, schedule, seed, reps);
                println!(
                    "{:16} {:12} seed{seed}  PE(A) {:.4} PE(B) {:.4} PE(novel) {:.4} (novelty_disc {:+.4}) | order_sens {:+.4} | regime_sep {:+.4} | {:.0} µs/step | calls[predict={} train={} ema={} perm={}]",
                    r.mode.to_string(),
                    r.schedule.to_string(),
                    r.pe_a,
                    r.pe_b,
                    r.pe_novel,
                    r.novelty_discrimination,
                    r.order_sensitivity,
                    r.regime_separation,
                    r.mean_step_us,
                    r.counts.hdc_ltc_predict,
                    r.counts.hdc_ltc_train,
                    r.counts.ema_updates,
                    r.counts.permutation_ops,
                );
                results.push(r);
            }
        }
        println!();
    }

    println!(
        "=== AGGREGATE (mean over {} pilot seeds) ===",
        pilot_seeds.len()
    );
    println!(
        "{:16} {:12} {:>10} {:>10} {:>14} {:>12} {:>12} {:>10}",
        "arm", "schedule", "PE(A)", "PE(B)", "novelty_disc", "order_sens", "regime_sep", "µs/step"
    );
    for &schedule in &schedules {
        for &mode in &modes {
            let rows: Vec<&RunResult> = results
                .iter()
                .filter(|r| r.mode == mode && r.schedule == schedule)
                .collect();
            let agg =
                |f: fn(&RunResult) -> f64| mean(&rows.iter().map(|r| f(r)).collect::<Vec<_>>());
            println!(
                "{:16} {:12} {:>10.4} {:>10.4} {:>14.4} {:>12.4} {:>12.4} {:>10.0}",
                mode.to_string(),
                schedule.to_string(),
                agg(|r| r.pe_a),
                agg(|r| r.pe_b),
                agg(|r| r.novelty_discrimination),
                agg(|r| r.order_sensitivity),
                agg(|r| r.regime_separation),
                agg(|r| r.mean_step_us),
            );
        }
    }

    println!(
        "\nPer plan §9: report per-arm, per-schedule, per-metric. No collapsed verdict is computed \
         here — read the aggregate table above against §9's acceptable-headline list before \
         drawing any conclusion. This is a BOUNDED PILOT (3 seeds) — do not treat as confirmatory; \
         plan §8 requires fresh seeds before any claim is finalized."
    );
}
