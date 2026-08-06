// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mixing chain: EQ → compressor → limiter for production-quality output.
//!
//! Applied as the final stage before output, ensuring clean, balanced audio
//! regardless of synthesis parameters.

/// Simple 3-band parametric EQ.
pub struct ParametricEQ {
    low_gain: f32,  // dB boost/cut at 200Hz
    mid_gain: f32,  // dB boost/cut at 1kHz
    high_gain: f32, // dB boost/cut at 5kHz
    low_state: [f32; 2],
    mid_state: [f32; 2],
    // No `high_state`: this is a complementary-filter crossover, so the high
    // band is derived by subtraction (`above_low - mid_band` in `process`)
    // and carries no filter state of its own.
}

impl ParametricEQ {
    pub fn new(low_db: f32, mid_db: f32, high_db: f32) -> Self {
        Self {
            low_gain: 10.0f32.powf(low_db / 20.0),
            mid_gain: 10.0f32.powf(mid_db / 20.0),
            high_gain: 10.0f32.powf(high_db / 20.0),
            low_state: [0.0; 2],
            mid_state: [0.0; 2],
        }
    }

    /// Flat EQ (no boost/cut).
    pub fn flat() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Process one sample through 3-band EQ.
    pub fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        // Low shelf (~200Hz)
        let low_alpha = 1.0 / (sample_rate / (std::f32::consts::TAU * 200.0) + 1.0);
        self.low_state[0] += low_alpha * (input - self.low_state[0]);
        let low_band = self.low_state[0];
        let above_low = input - low_band;

        // Mid band (~1kHz) extracted from above_low
        let mid_alpha = 1.0 / (sample_rate / (std::f32::consts::TAU * 1000.0) + 1.0);
        self.mid_state[0] += mid_alpha * (above_low - self.mid_state[0]);
        let mid_band = self.mid_state[0];
        let high_band = above_low - mid_band;

        // Recombine with gains
        low_band * self.low_gain + mid_band * self.mid_gain + high_band * self.high_gain
    }
}

/// Stereo compressor with sidechain from mid channel.
pub struct Compressor {
    threshold: f32, // linear amplitude
    ratio: f32,     // compression ratio (e.g., 4.0 = 4:1)
    attack_coeff: f32,
    release_coeff: f32,
    makeup_gain: f32, // linear
    envelope: f32,
}

impl Compressor {
    /// Create a compressor.
    /// - threshold_db: level above which compression engages (e.g., -12.0)
    /// - ratio: compression ratio (e.g., 3.0)
    /// - attack_ms, release_ms: envelope timing
    /// - makeup_db: gain applied after compression
    pub fn new(
        sample_rate: u32,
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_db: f32,
    ) -> Self {
        let sr = sample_rate as f32;
        Self {
            threshold: 10.0f32.powf(threshold_db / 20.0),
            ratio,
            attack_coeff: (-1.0 / (attack_ms * 0.001 * sr)).exp(),
            release_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
            makeup_gain: 10.0f32.powf(makeup_db / 20.0),
            envelope: 0.0,
        }
    }

    /// Gentle default: -14dB threshold, 2:1 ratio, 15ms attack, 150ms release, +2dB makeup.
    pub fn gentle(sample_rate: u32) -> Self {
        Self::new(sample_rate, -14.0, 2.0, 15.0, 150.0, 2.0)
    }

    /// Process stereo pair, returning compressed pair.
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let level = ((l * l + r * r) * 0.5).sqrt(); // RMS of stereo pair
        let coeff = if level > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * level;

        let gain = if self.envelope > self.threshold {
            let excess_db = 20.0 * (self.envelope / self.threshold).log10();
            let compressed_db = excess_db / self.ratio;
            10.0f32.powf(-(excess_db - compressed_db) / 20.0)
        } else {
            1.0
        };

        (l * gain * self.makeup_gain, r * gain * self.makeup_gain)
    }
}

/// Brick-wall limiter: prevents output from exceeding ±ceiling.
pub struct Limiter {
    ceiling: f32,
    release_coeff: f32,
    attenuation: f32,
}

impl Limiter {
    /// Create a limiter with ceiling in dB (e.g., -0.3 dB).
    pub fn new(sample_rate: u32, ceiling_db: f32) -> Self {
        let sr = sample_rate as f32;
        Self {
            ceiling: 10.0f32.powf(ceiling_db / 20.0),
            release_coeff: (-1.0 / (15.0 * 0.001 * sr)).exp(), // 15ms release (smoother)
            attenuation: 1.0,
        }
    }

    /// Default: -1.5 dB ceiling (safe headroom).
    pub fn default_limiter(sample_rate: u32) -> Self {
        Self::new(sample_rate, -1.5)
    }

    /// Process stereo pair.
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let peak = l.abs().max(r.abs());
        let target_atten = if peak > self.ceiling {
            self.ceiling / peak
        } else {
            1.0
        };

        // Instant attack, slow release
        if target_atten < self.attenuation {
            self.attenuation = target_atten;
        } else {
            self.attenuation =
                self.release_coeff * self.attenuation + (1.0 - self.release_coeff) * target_atten;
        }

        (l * self.attenuation, r * self.attenuation)
    }
}

/// Complete mixing chain: EQ → Compressor → Limiter.
pub struct MixingChain {
    pub eq_l: ParametricEQ,
    pub eq_r: ParametricEQ,
    pub compressor: Compressor,
    pub limiter: Limiter,
    sample_rate: f32,
}

impl MixingChain {
    /// Create a default mixing chain.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            eq_l: ParametricEQ::new(1.0, 0.0, -1.0), // gentle low boost, high cut
            eq_r: ParametricEQ::new(1.0, 0.0, -1.0),
            compressor: Compressor::gentle(sample_rate),
            limiter: Limiter::default_limiter(sample_rate),
            sample_rate: sample_rate as f32,
        }
    }

    /// Process a stereo sample through the full chain.
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        // EQ
        let eq_l = self.eq_l.process(l, self.sample_rate);
        let eq_r = self.eq_r.process(r, self.sample_rate);

        // Compressor
        let (comp_l, comp_r) = self.compressor.process(eq_l, eq_r);

        // Limiter
        self.limiter.process(comp_l, comp_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_flat_is_passthrough() {
        let mut eq = ParametricEQ::flat();
        let input = 0.5;
        // After a few samples to stabilize filters
        let mut out = 0.0;
        for _ in 0..100 {
            out = eq.process(input, 44100.0);
        }
        assert!(
            (out - input).abs() < 0.05,
            "flat EQ should pass through: {out}"
        );
    }

    #[test]
    fn compressor_reduces_loud_signals() {
        let mut comp = Compressor::gentle(44100);
        // Feed loud signal
        for _ in 0..100 {
            comp.process(0.8, 0.8);
        }
        let (l, r) = comp.process(0.8, 0.8);
        // With -12dB threshold (~0.25) and 3:1 ratio, 0.8 should be compressed
        // Makeup gain (+3dB ~1.41) partially compensates
        assert!(l < 0.8 * 1.5, "should compress loud signal: {l}");
        // It is a STEREO compressor driven by the RMS of the pair, so an
        // identical input on both channels must come out identically
        // compressed — checking only `l` would miss a gain applied to one
        // side, which is the failure mode that actually matters here.
        assert!(r < 0.8 * 1.5, "should compress the right channel too: {r}");
        assert!(
            (l - r).abs() < f32::EPSILON,
            "identical stereo input should compress identically: {l} vs {r}"
        );
    }

    #[test]
    fn limiter_prevents_clipping() {
        let mut lim = Limiter::default_limiter(44100);
        let (l, r) = lim.process(2.0, -2.0);
        assert!(l.abs() <= 1.0, "limiter should prevent clipping: {l}");
        assert!(r.abs() <= 1.0, "limiter should prevent clipping: {r}");
    }

    #[test]
    fn mixing_chain_all_finite() {
        let mut chain = MixingChain::new(44100);
        for i in 0..1000 {
            let input = (i as f32 * 0.1).sin() * 0.7;
            let (l, r) = chain.process(input, input);
            assert!(l.is_finite() && r.is_finite());
        }
    }

    #[test]
    fn mixing_chain_bounds_output() {
        let mut chain = MixingChain::new(44100);
        // Feed hot signal
        for _ in 0..500 {
            let (l, r) = chain.process(1.5, -1.5);
            assert!(l.abs() <= 1.5, "chain should bound output: {l}");
            assert!(r.abs() <= 1.5, "chain should bound output: {r}");
        }
    }
}
