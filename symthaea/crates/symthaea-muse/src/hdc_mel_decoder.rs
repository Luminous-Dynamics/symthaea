// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Baseline MLP decoder: `MusicalState → log-mel frame`.
//!
//! This is the Phase 2 **baseline** — a tiny pure-Rust MLP (17 → hidden → n_mels)
//! trained on MAESTRO `(state, mel_frame)` pairs via SGD. It exists to:
//! 1. Prove the data pipeline (load_pairs → train → loss decreases) end-to-end.
//! 2. Establish a reproducible baseline any future decoder must beat.
//! 3. Stay dep-free so iteration is fast (no CUDA / candle setup).
//!
//! When the baseline is healthy, the plan is to replace this with a
//! candle-backed CfC decoder trained on GPU (see `gpu_cfc.rs` in symthaea-broca).
//!
//! # Architecture
//! ```text
//!  state[17]  ──┐
//!               ▼
//!           Linear(17 → H) ──► ReLU ──► Linear(H → n_mels) ──► mel[n_mels]
//! ```
//!
//! Loss: mean-squared error between predicted and ground-truth log-mel frames.

use std::path::Path;

/// Decoder hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    pub state_dim: usize,
    pub hidden: usize,
    pub n_mels: usize,
    pub lr: f32,
    pub seed: u64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            state_dim: 17,
            hidden: 128,
            n_mels: 128,
            lr: 1e-3,
            seed: 0xBADC0FFEE,
        }
    }
}

/// Tiny MLP: state → hidden (ReLU) → mel. Manual forward/backward with SGD
/// or Adam (set via `enable_adam`).
pub struct MelDecoder {
    pub cfg: DecoderConfig,
    // Layer 1: state_dim × hidden
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    // Layer 2: hidden × n_mels
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,

    // Adam state (used iff `use_adam`)
    use_adam: bool,
    t: u64,
    w1_m: Vec<f32>, w1_v: Vec<f32>,
    b1_m: Vec<f32>, b1_v: Vec<f32>,
    w2_m: Vec<f32>, w2_v: Vec<f32>,
    b2_m: Vec<f32>, b2_v: Vec<f32>,

    // Persistent gradient buffers (reused across steps to avoid allocation).
    dw1: Vec<f32>, db1: Vec<f32>,
    dw2: Vec<f32>, db2: Vec<f32>,
    da1: Vec<f32>, dz1: Vec<f32>,
}

impl MelDecoder {
    /// Initialize with Glorot-uniform weights.
    pub fn new(cfg: DecoderConfig) -> Self {
        let mut rng = XorShift::new(cfg.seed);
        let w1_scale = (6.0f32 / (cfg.state_dim + cfg.hidden) as f32).sqrt();
        let w2_scale = (6.0f32 / (cfg.hidden + cfg.n_mels) as f32).sqrt();
        let w1 = (0..cfg.state_dim * cfg.hidden)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * w1_scale)
            .collect();
        let w2 = (0..cfg.hidden * cfg.n_mels)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * w2_scale)
            .collect();
        let b1 = vec![0.0; cfg.hidden];
        let b2 = vec![0.0; cfg.n_mels];
        let w1_m = vec![0.0; cfg.state_dim * cfg.hidden];
        let w1_v = vec![0.0; cfg.state_dim * cfg.hidden];
        let b1_m = vec![0.0; cfg.hidden];
        let b1_v = vec![0.0; cfg.hidden];
        let w2_m = vec![0.0; cfg.hidden * cfg.n_mels];
        let w2_v = vec![0.0; cfg.hidden * cfg.n_mels];
        let b2_m = vec![0.0; cfg.n_mels];
        let b2_v = vec![0.0; cfg.n_mels];
        let dw1 = vec![0.0; cfg.state_dim * cfg.hidden];
        let db1 = vec![0.0; cfg.hidden];
        let dw2 = vec![0.0; cfg.hidden * cfg.n_mels];
        let db2 = vec![0.0; cfg.n_mels];
        let da1 = vec![0.0; cfg.hidden];
        let dz1 = vec![0.0; cfg.hidden];
        Self {
            cfg, w1, b1, w2, b2,
            use_adam: false, t: 0,
            w1_m, w1_v, b1_m, b1_v, w2_m, w2_v, b2_m, b2_v,
            dw1, db1, dw2, db2, da1, dz1,
        }
    }

    /// Switch the optimizer to Adam. Call before `step()`.
    pub fn enable_adam(&mut self) { self.use_adam = true; }

    /// Forward pass. Returns (hidden_pre_relu, hidden_post_relu, mel_pred).
    pub fn forward(&self, state: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.cfg.hidden;
        let m = self.cfg.n_mels;
        let s = self.cfg.state_dim;

        // hidden = W1·state + b1
        let mut z1 = self.b1.clone();
        for j in 0..h {
            let mut acc = 0.0;
            for i in 0..s {
                acc += self.w1[i * h + j] * state[i];
            }
            z1[j] += acc;
        }
        // ReLU
        let a1: Vec<f32> = z1.iter().map(|&x| x.max(0.0)).collect();

        // mel = W2·a1 + b2
        let mut mel = self.b2.clone();
        for j in 0..m {
            let mut acc = 0.0;
            for i in 0..h {
                acc += self.w2[i * m + j] * a1[i];
            }
            mel[j] += acc;
        }

        (z1, a1, mel)
    }

    /// Predict-only forward (no intermediates).
    pub fn predict(&self, state: &[f32]) -> Vec<f32> {
        self.forward(state).2
    }

    /// Single SGD step on one (state, target_mel) pair.
    /// Returns the MSE loss for this example.
    pub fn step(&mut self, state: &[f32], target: &[f32]) -> f32 {
        let h = self.cfg.hidden;
        let m = self.cfg.n_mels;
        let s = self.cfg.state_dim;
        let lr = self.cfg.lr;

        let (z1, a1, mel) = self.forward(state);

        // Loss and output gradient: dL/dmel = 2 * (mel - target) / m
        let inv_m = 1.0 / m as f32;
        let mut dmel = vec![0.0f32; m];
        let mut loss = 0.0;
        for j in 0..m {
            let diff = mel[j] - target[j];
            loss += diff * diff;
            dmel[j] = 2.0 * diff * inv_m;
        }
        loss *= inv_m;

        // Gradient buffers live on self and are reused. Clear da1 in place.
        for i in 0..h { self.da1[i] = 0.0; }
        for i in 0..h {
            for j in 0..m {
                let idx = i * m + j;
                self.da1[i] += self.w2[idx] * dmel[j];
                self.dw2[idx] = dmel[j] * a1[i];
            }
        }
        for j in 0..m {
            self.db2[j] = dmel[j];
        }

        // ReLU backward
        for i in 0..h {
            self.dz1[i] = if z1[i] > 0.0 { self.da1[i] } else { 0.0 };
        }

        for i in 0..s {
            for j in 0..h {
                self.dw1[i * h + j] = self.dz1[j] * state[i];
            }
        }
        for j in 0..h {
            self.db1[j] = self.dz1[j];
        }

        if self.use_adam {
            self.t += 1;
            adam_update(&mut self.w1, &mut self.w1_m, &mut self.w1_v, &self.dw1, lr, self.t);
            adam_update(&mut self.b1, &mut self.b1_m, &mut self.b1_v, &self.db1, lr, self.t);
            adam_update(&mut self.w2, &mut self.w2_m, &mut self.w2_v, &self.dw2, lr, self.t);
            adam_update(&mut self.b2, &mut self.b2_m, &mut self.b2_v, &self.db2, lr, self.t);
        } else {
            for i in 0..self.dw1.len() { self.w1[i] -= lr * self.dw1[i]; }
            for i in 0..self.db1.len() { self.b1[i] -= lr * self.db1[i]; }
            for i in 0..self.dw2.len() { self.w2[i] -= lr * self.dw2[i]; }
            for i in 0..self.db2.len() { self.b2[i] -= lr * self.db2[i]; }
        }

        loss
    }

    /// Serialize weights to a flat binary file.
    ///
    /// Layout: 4 u32 header (state_dim, hidden, n_mels, 0), then w1, b1, w2, b2 as f32.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        for v in &[
            self.cfg.state_dim as u32,
            self.cfg.hidden as u32,
            self.cfg.n_mels as u32,
            0,
        ] {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in self.w1.iter().chain(self.b1.iter()).chain(self.w2.iter()).chain(self.b2.iter()) {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load weights from a file produced by `save`.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut header = [0u8; 16];
        f.read_exact(&mut header)?;
        let state_dim = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let n_mels = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let cfg = DecoderConfig { state_dim, hidden, n_mels, ..Default::default() };

        let read_vec = |f: &mut std::fs::File, n: usize| -> std::io::Result<Vec<f32>> {
            let mut buf = vec![0u8; n * 4];
            f.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let w1 = read_vec(&mut f, state_dim * hidden)?;
        let b1 = read_vec(&mut f, hidden)?;
        let w2 = read_vec(&mut f, hidden * n_mels)?;
        let b2 = read_vec(&mut f, n_mels)?;
        let w1_m = vec![0.0; state_dim * hidden];
        let w1_v = vec![0.0; state_dim * hidden];
        let b1_m = vec![0.0; hidden];
        let b1_v = vec![0.0; hidden];
        let w2_m = vec![0.0; hidden * n_mels];
        let w2_v = vec![0.0; hidden * n_mels];
        let b2_m = vec![0.0; n_mels];
        let b2_v = vec![0.0; n_mels];
        let dw1 = vec![0.0; state_dim * hidden];
        let db1 = vec![0.0; hidden];
        let dw2 = vec![0.0; hidden * n_mels];
        let db2 = vec![0.0; n_mels];
        let da1 = vec![0.0; hidden];
        let dz1 = vec![0.0; hidden];
        Ok(Self {
            cfg, w1, b1, w2, b2,
            use_adam: false, t: 0,
            w1_m, w1_v, b1_m, b1_v, w2_m, w2_v, b2_m, b2_v,
            dw1, db1, dw2, db2, da1, dz1,
        })
    }
}

/// Adam optimizer update. β1=0.9, β2=0.999, ε=1e-8.
fn adam_update(w: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32], lr: f32, t: u64) {
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    let eps = 1e-8f32;
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..w.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// Tiny xorshift RNG for deterministic init (no extra deps).
struct XorShift(u64);
impl XorShift {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEADBEEFCAFEBABE } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_shapes_match_config() {
        let cfg = DecoderConfig { state_dim: 17, hidden: 32, n_mels: 64, ..Default::default() };
        let dec = MelDecoder::new(cfg);
        assert_eq!(dec.w1.len(), 17 * 32);
        assert_eq!(dec.b1.len(), 32);
        assert_eq!(dec.w2.len(), 32 * 64);
        assert_eq!(dec.b2.len(), 64);
        let pred = dec.predict(&vec![0.5; 17]);
        assert_eq!(pred.len(), 64);
    }

    #[test]
    fn single_sample_overfits() {
        // A baseline sanity check: can we drive loss to ~0 on one example?
        let cfg = DecoderConfig { state_dim: 4, hidden: 16, n_mels: 8, lr: 0.1, seed: 1 };
        let mut dec = MelDecoder::new(cfg);
        let state = vec![0.3, -0.1, 0.7, 0.5];
        let target = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let initial = dec.step(&state, &target);
        for _ in 0..500 {
            dec.step(&state, &target);
        }
        let final_loss = dec.step(&state, &target);
        assert!(
            final_loss < initial * 0.01,
            "loss should drop 100x, initial={initial} final={final_loss}"
        );
    }

    #[test]
    fn adam_converges_faster_than_sgd_on_single_sample() {
        let cfg = DecoderConfig { state_dim: 4, hidden: 16, n_mels: 8, lr: 0.01, seed: 1 };
        let mut sgd = MelDecoder::new(cfg.clone());
        let mut adam = MelDecoder::new(cfg);
        adam.enable_adam();
        let state = vec![0.3, -0.1, 0.7, 0.5];
        let target = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let mut sgd_loss = 0.0;
        let mut adam_loss = 0.0;
        for _ in 0..100 {
            sgd_loss = sgd.step(&state, &target);
            adam_loss = adam.step(&state, &target);
        }
        assert!(
            adam_loss < sgd_loss,
            "Adam should converge faster at same lr; sgd={sgd_loss} adam={adam_loss}"
        );
    }

    #[test]
    fn save_load_roundtrip() {
        let cfg = DecoderConfig { state_dim: 4, hidden: 8, n_mels: 6, ..Default::default() };
        let mut dec = MelDecoder::new(cfg);
        // Perturb weights
        for _ in 0..10 {
            dec.step(&vec![0.1, 0.2, 0.3, 0.4], &vec![1.0; 6]);
        }
        let pred_before = dec.predict(&vec![0.5, 0.5, 0.5, 0.5]);
        let tmpfile = std::env::temp_dir().join("symthaea_decoder_test.bin");
        dec.save(&tmpfile).unwrap();
        let loaded = MelDecoder::load(&tmpfile).unwrap();
        let pred_after = loaded.predict(&vec![0.5, 0.5, 0.5, 0.5]);
        for (a, b) in pred_before.iter().zip(pred_after.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        std::fs::remove_file(&tmpfile).ok();
    }
}
