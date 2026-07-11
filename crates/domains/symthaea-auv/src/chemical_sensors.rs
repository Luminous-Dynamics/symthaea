// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chemical sensor simulation and anomaly detection.
//!
//! Simulates realistic sensor readings with noise, drift, and contamination plumes.
//! Detects anomalies by comparing readings to WHO thresholds + historical baselines.

use crate::types::ChemicalReadings;

/// Contamination plume: a localized region of elevated contaminant levels.
#[derive(Debug, Clone)]
pub struct ContaminationPlume {
    /// Center position [x, y, z].
    pub center: [f64; 3],
    /// Plume radius (meters).
    pub radius: f64,
    /// Which contaminant is elevated.
    pub contaminant: Contaminant,
    /// Peak concentration at center (in native units).
    pub peak_concentration: f64,
}

/// Contaminant types matching ChemicalReadings fields.
#[derive(Debug, Clone, Copy)]
pub enum Contaminant {
    Arsenic,
    Lead,
    Nitrates,
    Chlorine,
    LowOxygen,
    HighTurbidity,
    AcidicPh,
    HighTds,
}

/// Apply plume contamination to clean readings based on distance from plume center.
pub fn apply_plume(
    clean: &ChemicalReadings,
    plume: &ContaminationPlume,
    position: &[f64; 3],
) -> ChemicalReadings {
    let dx = position[0] - plume.center[0];
    let dy = position[1] - plume.center[1];
    let dz = position[2] - plume.center[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist > plume.radius {
        return *clean;
    }

    // Gaussian falloff from center
    let falloff = (-3.0 * (dist / plume.radius).powi(2)).exp();
    let added = plume.peak_concentration * falloff;

    let mut result = *clean;
    match plume.contaminant {
        Contaminant::Arsenic => result.arsenic_ug_l += added,
        Contaminant::Lead => result.lead_ug_l += added,
        Contaminant::Nitrates => result.nitrates_mg_l += added,
        Contaminant::Chlorine => result.chlorine_mg_l += added,
        Contaminant::LowOxygen => {
            result.dissolved_oxygen_mg_l = (result.dissolved_oxygen_mg_l - added).max(0.0)
        }
        Contaminant::HighTurbidity => result.turbidity_ntu += added,
        Contaminant::AcidicPh => result.ph = (result.ph - added).max(0.0),
        Contaminant::HighTds => result.tds_ppm += added,
    }
    result
}

/// Anomaly detection: returns list of contaminants exceeding WHO thresholds.
pub fn detect_anomalies(readings: &ChemicalReadings) -> Vec<String> {
    let mut anomalies = Vec::new();
    if readings.ph < 6.5 || readings.ph > 8.5 {
        anomalies.push(format!("pH: {:.1}", readings.ph));
    }
    if readings.turbidity_ntu > 5.0 {
        anomalies.push(format!("turbidity: {:.0} NTU", readings.turbidity_ntu));
    }
    if readings.tds_ppm > 1000.0 {
        anomalies.push(format!("TDS: {:.0} ppm", readings.tds_ppm));
    }
    if readings.nitrates_mg_l > 50.0 {
        anomalies.push(format!("nitrates: {:.1} mg/L", readings.nitrates_mg_l));
    }
    if readings.arsenic_ug_l > 10.0 {
        anomalies.push(format!("arsenic: {:.1} µg/L", readings.arsenic_ug_l));
    }
    if readings.lead_ug_l > 10.0 {
        anomalies.push(format!("lead: {:.1} µg/L", readings.lead_ug_l));
    }
    if readings.chlorine_mg_l > 5.0 {
        anomalies.push(format!("chlorine: {:.1} mg/L", readings.chlorine_mg_l));
    }
    if readings.dissolved_oxygen_mg_l < 4.0 {
        anomalies.push(format!("DO: {:.1} mg/L", readings.dissolved_oxygen_mg_l));
    }
    anomalies
}

// ── Sensor Noise Model ────────────────────────────────────────────────

/// Realistic sensor noise: additive Gaussian noise, mean-reverting drift,
/// and first-order temporal lag (IIR low-pass filter).
///
/// Each of the 8 chemical channels has independent noise/drift/lag.
#[derive(Debug, Clone)]
pub struct SensorNoiseModel {
    /// Gaussian noise standard deviation per channel.
    pub noise_std: [f64; 8],
    /// Drift rate per channel (random walk step size per dt).
    pub drift_rate: [f64; 8],
    /// Current drift state per channel.
    drift_state: [f64; 8],
    /// Time constant for first-order IIR lag (seconds).
    pub response_tau: f64,
    /// Filtered (lagged) output state.
    filtered: [f64; 8],
    /// Whether the model has been initialized with a first reading.
    initialized: bool,
}

impl SensorNoiseModel {
    /// Create a noise model with typical environmental sensor characteristics.
    pub fn typical() -> Self {
        Self {
            // Noise std: ~1% of typical range per channel
            noise_std: [0.05, 0.2, 5.0, 0.5, 0.5, 0.5, 0.05, 0.1],
            // Drift rate: slow random walk
            drift_rate: [0.001, 0.01, 0.5, 0.02, 0.02, 0.02, 0.002, 0.005],
            drift_state: [0.0; 8],
            response_tau: 2.0, // 2-second sensor lag
            filtered: [0.0; 8],
            initialized: false,
        }
    }

    /// Apply noise, drift, and temporal lag to clean sensor readings.
    ///
    /// `rng_seed` is incremented per call to produce deterministic but
    /// varying noise (avoids requiring `rand` as a non-dev dependency).
    pub fn apply(&mut self, clean: &ChemicalReadings, dt: f64, rng_seed: u64) -> ChemicalReadings {
        let channels = clean.to_array();
        let mut noisy = [0.0f64; 8];

        for i in 0..8 {
            // Deterministic pseudo-random noise (xorshift64)
            let mut s = rng_seed
                .wrapping_add(i as u64)
                .wrapping_mul(6364136223846793005);
            // Irwin-Hall Gaussian approximation: sum of 12 uniforms − 6
            // ≈ N(0,1) (mean 0, variance 1, near-Gaussian tails to ~3σ).
            // The previous single-uniform (u − 0.5)·√12 had the right mean
            // and variance but uniform (no) tails, which understates
            // anomaly-detector false-positive rates.
            let mut sum = 0.0f64;
            for _ in 0..12 {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                sum += (s.wrapping_mul(2685821657736338717) as f64) / (u64::MAX as f64);
            }
            let gaussian = sum - 6.0;

            // Additive noise
            let noise = gaussian * self.noise_std[i];

            // Mean-reverting drift: drift += rate * gaussian - revert * drift
            let revert = 0.01;
            self.drift_state[i] +=
                self.drift_rate[i] * gaussian * dt.sqrt() - revert * self.drift_state[i] * dt;

            // Clean + noise + drift
            noisy[i] = channels[i] + noise + self.drift_state[i];
        }

        // First-order IIR temporal lag: filtered = filtered + alpha * (noisy - filtered)
        let alpha = if self.response_tau > 0.0 {
            (dt / (self.response_tau + dt)).min(1.0)
        } else {
            1.0
        };

        if !self.initialized {
            self.filtered = noisy;
            self.initialized = true;
        } else {
            for i in 0..8 {
                self.filtered[i] += alpha * (noisy[i] - self.filtered[i]);
            }
        }

        ChemicalReadings::from_array(&self.filtered)
    }

    /// Reset drift and filter state.
    pub fn reset(&mut self) {
        self.drift_state = [0.0; 8];
        self.filtered = [0.0; 8];
        self.initialized = false;
    }
}

impl Default for SensorNoiseModel {
    fn default() -> Self {
        Self::typical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_water_no_anomalies() {
        let clean = ChemicalReadings::clean_freshwater();
        assert!(detect_anomalies(&clean).is_empty());
    }

    #[test]
    fn test_arsenic_plume() {
        let clean = ChemicalReadings::clean_freshwater();
        let plume = ContaminationPlume {
            center: [0.0, 0.0, 10.0],
            radius: 50.0,
            contaminant: Contaminant::Arsenic,
            peak_concentration: 40.0,
        };
        let at_center = apply_plume(&clean, &plume, &[0.0, 0.0, 10.0]);
        assert!(at_center.arsenic_ug_l > 10.0, "Should exceed WHO threshold");
        assert!(!detect_anomalies(&at_center).is_empty());

        let far_away = apply_plume(&clean, &plume, &[100.0, 0.0, 10.0]);
        assert!(
            (far_away.arsenic_ug_l - clean.arsenic_ug_l).abs() < 0.1,
            "Far from plume should be clean"
        );
    }

    #[test]
    fn test_noise_model_adds_variance() {
        let clean = ChemicalReadings::clean_freshwater();
        let mut model = SensorNoiseModel::typical();
        let mut readings = Vec::new();
        for i in 0..100 {
            readings.push(model.apply(&clean, 0.1, i as u64));
        }
        // pH should have nonzero variance (noise is being applied)
        let mean_ph: f64 = readings.iter().map(|r| r.ph).sum::<f64>() / 100.0;
        let var_ph: f64 = readings
            .iter()
            .map(|r| (r.ph - mean_ph).powi(2))
            .sum::<f64>()
            / 100.0;
        assert!(var_ph > 0.0, "Noise should add variance to readings");
    }

    #[test]
    fn test_noise_model_mean_near_clean() {
        let clean = ChemicalReadings::clean_freshwater();
        let mut model = SensorNoiseModel::typical();
        let mut ph_sum = 0.0;
        for i in 0..1000 {
            let r = model.apply(&clean, 0.1, i as u64);
            ph_sum += r.ph;
        }
        let mean_ph = ph_sum / 1000.0;
        // Mean should be close to clean value (within 0.5 for pH)
        assert!(
            (mean_ph - clean.ph).abs() < 0.5,
            "Mean pH {mean_ph} too far from clean {}",
            clean.ph
        );
    }

    #[test]
    fn test_noise_model_reset() {
        let clean = ChemicalReadings::clean_freshwater();
        let mut model = SensorNoiseModel::typical();
        // Run for a while to accumulate drift
        for i in 0..100 {
            model.apply(&clean, 0.1, i);
        }
        model.reset();
        assert!(!model.initialized);
    }

    #[test]
    fn test_noise_model_temporal_lag() {
        // With a large step, filter should track clean readings closely
        let clean = ChemicalReadings::clean_freshwater();
        let mut model = SensorNoiseModel::typical();
        model.noise_std = [0.0; 8]; // Disable noise to test lag only
        model.drift_rate = [0.0; 8]; // Disable drift

        // First reading initializes filter
        let r1 = model.apply(&clean, 10.0, 0); // Large dt → alpha ≈ 1.0
        assert!(
            (r1.ph - clean.ph).abs() < 0.01,
            "With large dt and no noise, should track clean"
        );
    }

    #[test]
    fn test_plume_gaussian_falloff() {
        let clean = ChemicalReadings::clean_freshwater();
        let plume = ContaminationPlume {
            center: [0.0, 0.0, 0.0],
            radius: 100.0,
            contaminant: Contaminant::Lead,
            peak_concentration: 50.0,
        };
        let close = apply_plume(&clean, &plume, &[10.0, 0.0, 0.0]);
        let mid = apply_plume(&clean, &plume, &[50.0, 0.0, 0.0]);
        assert!(
            close.lead_ug_l > mid.lead_ug_l,
            "Closer should have higher concentration"
        );
    }
}
