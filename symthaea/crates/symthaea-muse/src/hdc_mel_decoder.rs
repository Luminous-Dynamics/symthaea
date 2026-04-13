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
#[derive(Debug, Clone)]
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

/// Tiny MLP: state → hidden (ReLU) → mel. Manual forward/backward with SGD.
pub struct MelDecoder {
    pub cfg: DecoderConfig,
    // Layer 1: state_dim × hidden
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    // Layer 2: hidden × n_mels
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
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
        Self { cfg, w1, b1, w2, b2 }
    }

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

        // Backprop into W2, b2, and a1
        let mut da1 = vec![0.0f32; h];
        for i in 0..h {
            for j in 0..m {
                let idx = i * m + j;
                da1[i] += self.w2[idx] * dmel[j];
                self.w2[idx] -= lr * dmel[j] * a1[i];
            }
        }
        for j in 0..m {
            self.b2[j] -= lr * dmel[j];
        }

        // ReLU backward
        let mut dz1 = vec![0.0f32; h];
        for i in 0..h {
            if z1[i] > 0.0 {
                dz1[i] = da1[i];
            }
        }

        // Backprop into W1, b1
        for i in 0..s {
            for j in 0..h {
                self.w1[i * h + j] -= lr * dz1[j] * state[i];
            }
        }
        for j in 0..h {
            self.b1[j] -= lr * dz1[j];
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
        Ok(Self { cfg, w1, b1, w2, b2 })
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
