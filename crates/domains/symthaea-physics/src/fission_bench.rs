// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fission anomaly-detection benchmark — ground-truth gated.
//!
//! Extends the 5-channel [`crate::fission`] demo toward a defensible
//! monitoring result: a realistic 21-channel PWR sensor suite, a
//! deterministic signature-level plant simulator, a fault-injection library
//! with known onset times, and a calibrated free-energy detector scored on
//! **detection latency and false-alarm rate against ground truth** — not on
//! HDC self-similarity.
//!
//! ## Honesty notes
//! - The simulator reproduces *fault signatures* (which channels move, in
//!   which direction, on what timescale), not thermal-hydraulics. External
//!   dataset validation is the next step (see
//!   `symthaea/NUCLEAR_ENERGY_PLAN_2026-07-06.md`, Phase 1).
//! - This is advisory (non-1E) monitoring tooling. It is not, and must never
//!   be presented as, a safety-grade reactor protection function.
//! - Known limitation, kept deliberately: a stuck sensor at steady state is
//!   *not detectable* by a reference-state detector (the frozen value looks
//!   healthy). See `test_stuck_sensor_undetected_at_steady_state`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Sensor suite ────────────────────────────────────────────────────────

/// Number of monitored plant channels.
pub const PLANT_CHANNELS: usize = 21;

/// Channel indices (into the normalized vector). Grouped: primary loop,
/// neutronics, secondary side, support systems.
pub mod channel {
    pub const HOT_LEG_TEMP: usize = 0;
    pub const COLD_LEG_TEMP: usize = 1;
    pub const PRIMARY_PRESSURE: usize = 2;
    pub const PRESSURIZER_LEVEL: usize = 3;
    pub const PRIMARY_FLOW: usize = 4;
    pub const PUMP_SPEED: usize = 5;
    pub const NEUTRON_FLUX_POWER: usize = 6;
    pub const NEUTRON_FLUX_INTERMEDIATE: usize = 7;
    pub const CONTROL_ROD_POS: usize = 8;
    pub const BORON_PPM: usize = 9;
    pub const SG_LEVEL: usize = 10;
    pub const SG_PRESSURE: usize = 11;
    pub const FEEDWATER_FLOW: usize = 12;
    pub const STEAM_FLOW: usize = 13;
    pub const TURBINE_POWER: usize = 14;
    pub const CONTAINMENT_PRESSURE: usize = 15;
    pub const CONTAINMENT_TEMP: usize = 16;
    pub const COOLANT_ACTIVITY: usize = 17;
    pub const PUMP_VIBRATION: usize = 18;
    pub const CHARGING_FLOW: usize = 19;
    pub const LETDOWN_TEMP: usize = 20;
}

/// Human-readable channel labels, index-aligned with [`channel`].
pub const PLANT_CHANNEL_LABELS: [&str; PLANT_CHANNELS] = [
    "hot leg temp (°C/350)",
    "cold leg temp (°C/320)",
    "primary pressure (MPa/17)",
    "pressurizer level (frac)",
    "primary flow (frac rated)",
    "RCP speed (frac rated)",
    "neutron flux power-range (frac)",
    "neutron flux intermediate (frac)",
    "control rod insertion (frac)",
    "boron (ppm/2000)",
    "SG level (frac)",
    "SG pressure (MPa/8)",
    "feedwater flow (frac rated)",
    "steam flow (frac rated)",
    "turbine power (frac rated)",
    "containment pressure (kPa/200)",
    "containment temp (°C/80)",
    "coolant activity (frac scale)",
    "RCP vibration (mm/s / 10)",
    "charging flow (frac rated)",
    "letdown temp (°C/60)",
];

/// A full plant sensor reading in raw engineering units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantReading {
    pub hot_leg_temp_c: f64,
    pub cold_leg_temp_c: f64,
    pub primary_pressure_mpa: f64,
    pub pressurizer_level: f64,
    pub primary_flow: f64,
    pub pump_speed: f64,
    pub neutron_flux_power: f64,
    pub neutron_flux_intermediate: f64,
    pub control_rod_pos: f64,
    pub boron_ppm: f64,
    pub sg_level: f64,
    pub sg_pressure_mpa: f64,
    pub feedwater_flow: f64,
    pub steam_flow: f64,
    pub turbine_power: f64,
    pub containment_pressure_kpa: f64,
    pub containment_temp_c: f64,
    pub coolant_activity: f64,
    pub pump_vibration_mm_s: f64,
    pub charging_flow: f64,
    pub letdown_temp_c: f64,
}

impl PlantReading {
    /// Normalize every channel into [0, 1] for HDC encoding.
    pub fn to_normalized(&self) -> [f64; PLANT_CHANNELS] {
        let mut v = [0.0; PLANT_CHANNELS];
        v[channel::HOT_LEG_TEMP] = self.hot_leg_temp_c / 350.0;
        v[channel::COLD_LEG_TEMP] = self.cold_leg_temp_c / 320.0;
        v[channel::PRIMARY_PRESSURE] = self.primary_pressure_mpa / 17.0;
        v[channel::PRESSURIZER_LEVEL] = self.pressurizer_level;
        v[channel::PRIMARY_FLOW] = self.primary_flow;
        v[channel::PUMP_SPEED] = self.pump_speed;
        v[channel::NEUTRON_FLUX_POWER] = self.neutron_flux_power;
        v[channel::NEUTRON_FLUX_INTERMEDIATE] = self.neutron_flux_intermediate;
        v[channel::CONTROL_ROD_POS] = self.control_rod_pos;
        v[channel::BORON_PPM] = self.boron_ppm / 2000.0;
        v[channel::SG_LEVEL] = self.sg_level;
        v[channel::SG_PRESSURE] = self.sg_pressure_mpa / 8.0;
        v[channel::FEEDWATER_FLOW] = self.feedwater_flow;
        v[channel::STEAM_FLOW] = self.steam_flow;
        v[channel::TURBINE_POWER] = self.turbine_power;
        v[channel::CONTAINMENT_PRESSURE] = self.containment_pressure_kpa / 200.0;
        v[channel::CONTAINMENT_TEMP] = self.containment_temp_c / 80.0;
        v[channel::COOLANT_ACTIVITY] = self.coolant_activity;
        v[channel::PUMP_VIBRATION] = self.pump_vibration_mm_s / 10.0;
        v[channel::CHARGING_FLOW] = self.charging_flow;
        v[channel::LETDOWN_TEMP] = self.letdown_temp_c / 60.0;
        for x in &mut v {
            *x = x.clamp(0.0, 1.0);
        }
        v
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────

/// HDC encoder for the 21-channel plant state.
pub struct PlantHdcEncoder {
    bases: Vec<ContinuousHV>,
}

impl PlantHdcEncoder {
    pub fn new() -> Self {
        let bases = (0..PLANT_CHANNELS)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, 0xF15_2000 + i as u64))
            .collect();
        Self { bases }
    }

    pub fn encode(&self, normalized: &[f64; PLANT_CHANNELS]) -> ContinuousHV {
        let weights: Vec<f32> = normalized
            .iter()
            .map(|&x| x.clamp(0.0, 1.0) as f32)
            .collect();
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for PlantHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fault library ───────────────────────────────────────────────────────

/// Injectable plant faults with physically-motivated channel signatures.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FaultKind {
    /// A sensor reads progressively wrong: normalized value drifts at
    /// `rate_per_s` (plant itself stays healthy).
    SensorDrift { channel: usize, rate_per_s: f64 },
    /// A sensor freezes at its onset value (plant stays healthy).
    StuckSensor { channel: usize },
    /// Reactor coolant pump coasts down with time constant `tau_s`:
    /// pump speed and flow decay, hot leg heats up, vibration rises.
    PumpCoastdown { tau_s: f64 },
    /// Small-break LOCA: primary pressure and pressurizer level fall,
    /// containment pressure/temp/activity rise, charging (makeup) rises.
    SmallBreakLoca { depress_mpa_per_s: f64 },
    /// Feedwater controller oscillation: feedwater flow and SG level
    /// oscillate with `amplitude` (normalized) and `period_s`.
    FeedwaterOscillation { amplitude: f64, period_s: f64 },
    /// Dropped control rod: step insertion, flux/power fall promptly,
    /// turbine lags, hot leg cools.
    ControlRodDrop,
}

/// A fault plus its ground-truth onset time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaultScenario {
    pub kind: FaultKind,
    pub onset_s: f64,
}

// ── Simulator ───────────────────────────────────────────────────────────

/// Deterministic signature-level PWR simulator.
///
/// Produces steady-state operation at a given power fraction with seeded
/// sensor noise, and applies at most one [`FaultScenario`]'s channel
/// signatures once its onset time passes. This is NOT a thermal-hydraulics
/// code; it exists to give the detector labeled, physically-plausible
/// fault signatures.
pub struct PlantSimulator {
    power: f64,
    noise_amp: f64,
    rng: ChaCha8Rng,
    t: f64,
    fault: Option<FaultScenario>,
    stuck_value: Option<f64>,
    last_reading: Option<PlantReading>,
}

impl PlantSimulator {
    pub fn new(power_fraction: f64, seed: u64) -> Self {
        Self {
            power: power_fraction.clamp(0.0, 1.0),
            noise_amp: 0.002,
            rng: ChaCha8Rng::seed_from_u64(seed),
            t: 0.0,
            fault: None,
            stuck_value: None,
            last_reading: None,
        }
    }

    pub fn inject(&mut self, scenario: FaultScenario) {
        self.fault = Some(scenario);
        self.stuck_value = None;
    }

    pub fn time(&self) -> f64 {
        self.t
    }

    /// Raw reading from the most recent [`Self::step`] (before sensor-level
    /// fault corruption), for signature inspection in tests.
    pub fn last_reading(&self) -> Option<&PlantReading> {
        self.last_reading.as_ref()
    }

    /// Healthy steady-state reading at power fraction `p` (no noise).
    pub fn steady_state(p: f64) -> PlantReading {
        PlantReading {
            hot_leg_temp_c: 292.0 + 34.0 * p,
            cold_leg_temp_c: 286.0 + 6.0 * p,
            primary_pressure_mpa: 15.5,
            pressurizer_level: 0.55,
            primary_flow: 1.0,
            pump_speed: 1.0,
            neutron_flux_power: p,
            neutron_flux_intermediate: p,
            control_rod_pos: 0.25 - 0.15 * p,
            boron_ppm: 900.0,
            sg_level: 0.6,
            sg_pressure_mpa: 6.9,
            feedwater_flow: p,
            steam_flow: p,
            turbine_power: p,
            containment_pressure_kpa: 101.0,
            containment_temp_c: 40.0,
            coolant_activity: 0.05,
            pump_vibration_mm_s: 2.0,
            charging_flow: 0.5,
            letdown_temp_c: 45.0,
        }
    }

    /// Advance `dt` seconds; return the (possibly sensor-corrupted)
    /// normalized channel vector.
    pub fn step(&mut self, dt: f64) -> [f64; PLANT_CHANNELS] {
        assert!(dt.is_finite() && dt > 0.0);
        self.t += dt;
        let mut r = Self::steady_state(self.power);

        // Physical fault signatures (plant-side).
        let mut sensor_fault: Option<(FaultKind, f64)> = None;
        if let Some(sc) = self.fault {
            let ft = self.t - sc.onset_s;
            if ft >= 0.0 {
                match sc.kind {
                    FaultKind::PumpCoastdown { tau_s } => {
                        let decay = (-ft / tau_s).exp();
                        r.pump_speed *= decay;
                        r.primary_flow *= decay;
                        // Reduced heat removal: hot leg heats with lost flow.
                        r.hot_leg_temp_c += 30.0 * self.power * (1.0 - decay);
                        r.cold_leg_temp_c += 8.0 * self.power * (1.0 - decay);
                        // Bearing rumble during coastdown, settling as it stops.
                        r.pump_vibration_mm_s += 4.0 * (1.0 - decay) * decay.max(0.1);
                    }
                    FaultKind::SmallBreakLoca { depress_mpa_per_s } => {
                        let dp = depress_mpa_per_s * ft;
                        r.primary_pressure_mpa = (r.primary_pressure_mpa - dp).max(7.0);
                        r.pressurizer_level = (r.pressurizer_level - 0.02 * dp / 0.1).max(0.05);
                        r.charging_flow = (r.charging_flow + 0.05 * dp / 0.1).min(1.0);
                        r.containment_pressure_kpa += 40.0 * (1.0 - (-ft / 300.0).exp());
                        r.containment_temp_c += 15.0 * (1.0 - (-ft / 300.0).exp());
                        r.coolant_activity = (r.coolant_activity + 0.001 * ft).min(1.0);
                    }
                    FaultKind::FeedwaterOscillation {
                        amplitude,
                        period_s,
                    } => {
                        let phase = 2.0 * std::f64::consts::PI * ft / period_s;
                        r.feedwater_flow =
                            (r.feedwater_flow + amplitude * phase.sin()).clamp(0.0, 1.0);
                        // SG level integrates the mismatch, lagging ~quarter period.
                        r.sg_level = (r.sg_level
                            + amplitude * 0.8 * (phase - std::f64::consts::FRAC_PI_2).sin())
                        .clamp(0.0, 1.0);
                        r.steam_flow =
                            (r.steam_flow + amplitude * 0.3 * phase.sin()).clamp(0.0, 1.0);
                    }
                    FaultKind::ControlRodDrop => {
                        let resp = 1.0 - (-ft / 5.0).exp(); // prompt, ~5s
                        r.control_rod_pos = (r.control_rod_pos + 0.3 * resp).min(1.0);
                        r.neutron_flux_power = (r.neutron_flux_power - 0.25 * resp).max(0.0);
                        r.neutron_flux_intermediate =
                            (r.neutron_flux_intermediate - 0.25 * resp).max(0.0);
                        // Thermal side lags (~60s).
                        let lag = 1.0 - (-ft / 60.0).exp();
                        r.turbine_power = (r.turbine_power - 0.25 * lag).max(0.0);
                        r.hot_leg_temp_c -= 10.0 * lag;
                        r.steam_flow = (r.steam_flow - 0.2 * lag).max(0.0);
                    }
                    // Sensor-level faults handled after normalization.
                    k @ (FaultKind::SensorDrift { .. } | FaultKind::StuckSensor { .. }) => {
                        sensor_fault = Some((k, ft));
                    }
                }
            }
        }

        self.last_reading = Some(r.clone());
        let mut v = r.to_normalized();

        // Seeded sensor noise on every channel.
        for x in &mut v {
            *x = (*x + (self.rng.r#gen::<f64>() - 0.5) * 2.0 * self.noise_amp).clamp(0.0, 1.0);
        }

        // Sensor-level corruption.
        if let Some((kind, ft)) = sensor_fault {
            match kind {
                FaultKind::SensorDrift {
                    channel,
                    rate_per_s,
                } => {
                    v[channel] = (v[channel] + rate_per_s * ft).clamp(0.0, 1.0);
                }
                FaultKind::StuckSensor { channel } => {
                    let frozen = *self.stuck_value.get_or_insert(v[channel]);
                    v[channel] = frozen;
                }
                _ => unreachable!(),
            }
        }

        v
    }
}

// ── Detector ────────────────────────────────────────────────────────────

/// Calibrated free-energy anomaly detector.
///
/// Encodes each reading, computes free energy against a healthy reference
/// state, calibrates mean/std on a healthy window, then alarms when FE
/// exceeds `mean + k_sigma * std` for `persistence` consecutive samples.
pub struct FreeEnergyDetector {
    encoder: PlantHdcEncoder,
    reference: ContinuousHV,
    k_sigma: f64,
    persistence: usize,
    baseline_mean: f64,
    baseline_std: f64,
    consecutive: usize,
    calibrated: bool,
}

impl FreeEnergyDetector {
    pub fn new(reference_state: &[f64; PLANT_CHANNELS], k_sigma: f64, persistence: usize) -> Self {
        let encoder = PlantHdcEncoder::new();
        let reference = encoder.encode(reference_state);
        Self {
            encoder,
            reference,
            k_sigma,
            persistence,
            baseline_mean: 0.0,
            baseline_std: 0.0,
            consecutive: 0,
            calibrated: false,
        }
    }

    pub fn free_energy(&self, normalized: &[f64; PLANT_CHANNELS]) -> f64 {
        let hv = self.encoder.encode(normalized);
        let sim = hv.similarity(&self.reference) as f64;
        if !sim.is_finite() {
            1.0
        } else {
            (1.0 - sim).max(0.0)
        }
    }

    /// Fit baseline statistics from healthy samples.
    pub fn calibrate(&mut self, healthy: &[[f64; PLANT_CHANNELS]]) {
        assert!(healthy.len() >= 10, "need >=10 calibration samples");
        let fes: Vec<f64> = healthy.iter().map(|s| self.free_energy(s)).collect();
        let n = fes.len() as f64;
        let mean = fes.iter().sum::<f64>() / n;
        let var = fes.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
        self.baseline_mean = mean;
        self.baseline_std = var.sqrt().max(1e-9);
        self.consecutive = 0;
        self.calibrated = true;
    }

    pub fn threshold(&self) -> f64 {
        self.baseline_mean + self.k_sigma * self.baseline_std
    }

    /// Feed one sample; returns true when the alarm fires (persistence met).
    pub fn observe(&mut self, normalized: &[f64; PLANT_CHANNELS]) -> bool {
        assert!(self.calibrated, "calibrate() before observe()");
        if self.free_energy(normalized) > self.threshold() {
            self.consecutive += 1;
        } else {
            self.consecutive = 0;
        }
        self.consecutive >= self.persistence
    }
}

// ── Benchmark ───────────────────────────────────────────────────────────

/// Ground-truth outcome for one injected fault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub kind: FaultKind,
    pub onset_s: f64,
    pub detected: bool,
    /// Alarm time minus ground-truth onset (present only when detected).
    pub detection_latency_s: Option<f64>,
    /// Alarms raised before the fault onset (false positives).
    pub false_alarms_before_onset: usize,
}

/// Aggregate benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub scenarios: Vec<ScenarioResult>,
    pub detected: usize,
    pub total: usize,
    pub mean_latency_s: f64,
    /// False alarms per hour measured on a separate healthy run.
    pub healthy_false_alarms_per_hour: f64,
}

/// Detector configuration used by [`run_benchmark`].
pub const BENCH_K_SIGMA: f64 = 8.0;
pub const BENCH_PERSISTENCE: usize = 3;
const BENCH_DT_S: f64 = 1.0;
const BENCH_CALIBRATION_S: f64 = 300.0;
const BENCH_POST_ONSET_S: f64 = 600.0;
const BENCH_HEALTHY_S: f64 = 3_600.0;

/// Run one fault scenario against a freshly calibrated detector.
pub fn run_scenario(kind: FaultKind, power: f64, seed: u64) -> ScenarioResult {
    let onset_s = BENCH_CALIBRATION_S + 100.0;
    let mut sim = PlantSimulator::new(power, seed);
    sim.inject(FaultScenario { kind, onset_s });

    // Calibration window (pre-onset, healthy by construction).
    let mut calib = Vec::new();
    while sim.time() < BENCH_CALIBRATION_S {
        calib.push(sim.step(BENCH_DT_S));
    }
    let reference = PlantSimulator::steady_state(power).to_normalized();
    let mut det = FreeEnergyDetector::new(&reference, BENCH_K_SIGMA, BENCH_PERSISTENCE);
    det.calibrate(&calib);

    let mut false_alarms_before_onset = 0;
    let mut detection_latency_s = None;
    while sim.time() < onset_s + BENCH_POST_ONSET_S {
        let v = sim.step(BENCH_DT_S);
        if det.observe(&v) {
            if sim.time() < onset_s {
                false_alarms_before_onset += 1;
            } else {
                detection_latency_s = Some(sim.time() - onset_s);
                break;
            }
        }
    }

    ScenarioResult {
        kind,
        onset_s,
        detected: detection_latency_s.is_some(),
        detection_latency_s,
        false_alarms_before_onset,
    }
}

/// Measure false alarms per hour on a healthy (fault-free) run.
pub fn run_healthy_false_alarm_check(power: f64, seed: u64) -> f64 {
    let mut sim = PlantSimulator::new(power, seed);
    let mut calib = Vec::new();
    while sim.time() < BENCH_CALIBRATION_S {
        calib.push(sim.step(BENCH_DT_S));
    }
    let reference = PlantSimulator::steady_state(power).to_normalized();
    let mut det = FreeEnergyDetector::new(&reference, BENCH_K_SIGMA, BENCH_PERSISTENCE);
    det.calibrate(&calib);

    let mut alarms = 0usize;
    let start = sim.time();
    while sim.time() < start + BENCH_HEALTHY_S {
        let v = sim.step(BENCH_DT_S);
        if det.observe(&v) {
            alarms += 1;
        }
    }
    alarms as f64 / (BENCH_HEALTHY_S / 3_600.0)
}

/// The standard detectable-fault suite (excludes the documented
/// stuck-sensor-at-steady-state blind spot).
pub fn standard_fault_suite() -> Vec<FaultKind> {
    vec![
        FaultKind::PumpCoastdown { tau_s: 60.0 },
        FaultKind::SmallBreakLoca {
            depress_mpa_per_s: 0.005,
        },
        FaultKind::FeedwaterOscillation {
            amplitude: 0.08,
            period_s: 40.0,
        },
        FaultKind::ControlRodDrop,
        FaultKind::SensorDrift {
            channel: channel::HOT_LEG_TEMP,
            rate_per_s: 0.0005,
        },
    ]
}

/// Full benchmark: every standard fault plus a healthy false-alarm run.
pub fn run_benchmark(power: f64, seed: u64) -> BenchmarkReport {
    let scenarios: Vec<ScenarioResult> = standard_fault_suite()
        .into_iter()
        .enumerate()
        .map(|(i, kind)| run_scenario(kind, power, seed.wrapping_add(i as u64)))
        .collect();
    let detected = scenarios.iter().filter(|s| s.detected).count();
    let latencies: Vec<f64> = scenarios
        .iter()
        .filter_map(|s| s.detection_latency_s)
        .collect();
    let mean_latency_s = if latencies.is_empty() {
        f64::NAN
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    BenchmarkReport {
        total: scenarios.len(),
        detected,
        mean_latency_s,
        healthy_false_alarms_per_hour: run_healthy_false_alarm_check(power, seed ^ 0xDEAD),
        scenarios,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const POWER: f64 = 0.9;
    const SEED: u64 = 0xF15_5EED;

    #[test]
    fn test_steady_state_normalized_in_bounds() {
        let v = PlantSimulator::steady_state(POWER).to_normalized();
        for (i, x) in v.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(x) && *x > 0.0,
                "channel {} ({}) out of bounds: {}",
                i,
                PLANT_CHANNEL_LABELS[i],
                x
            );
        }
    }

    #[test]
    fn test_channel_labels_count() {
        assert_eq!(PLANT_CHANNEL_LABELS.len(), PLANT_CHANNELS);
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = PlantHdcEncoder::new();
        let hv = enc.encode(&PlantSimulator::steady_state(POWER).to_normalized());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_simulator_deterministic() {
        let mut a = PlantSimulator::new(POWER, SEED);
        let mut b = PlantSimulator::new(POWER, SEED);
        for _ in 0..50 {
            assert_eq!(a.step(1.0), b.step(1.0));
        }
    }

    #[test]
    fn test_pump_coastdown_signature() {
        let mut sim = PlantSimulator::new(POWER, SEED);
        sim.inject(FaultScenario {
            kind: FaultKind::PumpCoastdown { tau_s: 60.0 },
            onset_s: 10.0,
        });
        for _ in 0..200 {
            sim.step(1.0);
        }
        let r = sim.last_reading().unwrap();
        let healthy = PlantSimulator::steady_state(POWER);
        assert!(r.pump_speed < 0.1, "pump should have coasted down");
        assert!(r.primary_flow < 0.1);
        assert!(r.hot_leg_temp_c > healthy.hot_leg_temp_c + 20.0);
    }

    #[test]
    fn test_loca_signature() {
        let mut sim = PlantSimulator::new(POWER, SEED);
        sim.inject(FaultScenario {
            kind: FaultKind::SmallBreakLoca {
                depress_mpa_per_s: 0.005,
            },
            onset_s: 10.0,
        });
        for _ in 0..400 {
            sim.step(1.0);
        }
        let r = sim.last_reading().unwrap();
        let healthy = PlantSimulator::steady_state(POWER);
        assert!(r.primary_pressure_mpa < healthy.primary_pressure_mpa - 1.0);
        assert!(r.containment_pressure_kpa > healthy.containment_pressure_kpa + 10.0);
        assert!(r.pressurizer_level < healthy.pressurizer_level);
    }

    #[test]
    fn test_rod_drop_signature() {
        let mut sim = PlantSimulator::new(POWER, SEED);
        sim.inject(FaultScenario {
            kind: FaultKind::ControlRodDrop,
            onset_s: 10.0,
        });
        for _ in 0..120 {
            sim.step(1.0);
        }
        let r = sim.last_reading().unwrap();
        assert!(r.neutron_flux_power < POWER - 0.2);
        assert!(r.control_rod_pos > 0.25 - 0.15 * POWER + 0.25);
    }

    #[test]
    fn test_detector_calibration_threshold_positive() {
        let mut sim = PlantSimulator::new(POWER, SEED);
        let samples: Vec<_> = (0..100).map(|_| sim.step(1.0)).collect();
        let reference = PlantSimulator::steady_state(POWER).to_normalized();
        let mut det = FreeEnergyDetector::new(&reference, BENCH_K_SIGMA, BENCH_PERSISTENCE);
        det.calibrate(&samples);
        assert!(det.threshold() > 0.0 && det.threshold().is_finite());
    }

    #[test]
    fn test_detects_pump_coastdown() {
        let res = run_scenario(FaultKind::PumpCoastdown { tau_s: 60.0 }, POWER, SEED);
        assert!(res.detected, "pump coastdown missed: {:?}", res);
        assert!(
            res.detection_latency_s.unwrap() < 120.0,
            "latency too high: {:?}",
            res
        );
        assert_eq!(res.false_alarms_before_onset, 0);
    }

    #[test]
    fn test_detects_loca() {
        let res = run_scenario(
            FaultKind::SmallBreakLoca {
                depress_mpa_per_s: 0.005,
            },
            POWER,
            SEED,
        );
        assert!(res.detected, "LOCA missed: {:?}", res);
        assert!(
            res.detection_latency_s.unwrap() < 400.0,
            "latency too high: {:?}",
            res
        );
    }

    #[test]
    fn test_detects_rod_drop() {
        let res = run_scenario(FaultKind::ControlRodDrop, POWER, SEED);
        assert!(res.detected, "rod drop missed: {:?}", res);
        assert!(
            res.detection_latency_s.unwrap() < 60.0,
            "latency too high: {:?}",
            res
        );
    }

    #[test]
    fn test_detects_sensor_drift_eventually() {
        let res = run_scenario(
            FaultKind::SensorDrift {
                channel: channel::HOT_LEG_TEMP,
                rate_per_s: 0.0005,
            },
            POWER,
            SEED,
        );
        assert!(res.detected, "sensor drift missed: {:?}", res);
    }

    #[test]
    fn test_stuck_sensor_undetected_at_steady_state() {
        // Documented blind spot of a reference-state detector: a frozen
        // sensor at steady state looks healthy. This test pins the
        // limitation so a future detector upgrade flips it consciously.
        let res = run_scenario(
            FaultKind::StuckSensor {
                channel: channel::HOT_LEG_TEMP,
            },
            POWER,
            SEED,
        );
        assert!(
            !res.detected,
            "stuck sensor unexpectedly detected — detector improved? Update the plan + this test."
        );
    }

    #[test]
    fn test_healthy_false_alarm_rate_low() {
        let rate = run_healthy_false_alarm_check(POWER, SEED);
        assert!(rate <= 1.0, "false alarms per hour too high: {}", rate);
    }

    #[test]
    fn test_full_benchmark_ground_truth_gate() {
        let report = run_benchmark(POWER, SEED);
        assert_eq!(report.total, 5);
        assert!(
            report.detected >= 4,
            "expected >=4/5 faults detected, got {}/{}: {:?}",
            report.detected,
            report.total,
            report.scenarios
        );
        assert!(report.healthy_false_alarms_per_hour <= 1.0);
    }
}
