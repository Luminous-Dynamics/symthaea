// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! External validation of the [`crate::fission_bench`] detection approach
//! against real PWR-simulator data.
//!
//! `fission_bench` proved the calibrated-free-energy detector on a
//! signature-level simulator we wrote ourselves. This module asks the
//! harder question: does the same approach (HDC encode → free energy vs a
//! healthy reference → calibrated kσ/persistence alarm) generalize to
//! transients from an independent, real PWR training simulator?
//!
//! Data: [NPPAD](https://github.com/thu-inet/NuclearPowerPlantAccidentData)
//! (MIT licensed, PCTRAN-simulated), fixtures embedded from
//! `tests/fixtures/nppad/` — see that directory's `README.md` for
//! provenance and per-case ground truth (accident onset time, and the
//! automatic-scram time where one occurs).
//!
//! ## Result (measured 2026-07-07, see the pinned tests below)
//! The answer is a qualified no, and the qualification is the finding:
//! - On the three cases with an automatic reactor scram (LOCA 1%, LOCA
//!   50%, rod withdrawal), the detector fires reliably but **always
//!   shortly *after* the plant's own hardwired trip** (lag ~7-13s across
//!   3σ/persist-1 through 6σ/persist-2 tuning), never before. Diagnostic
//!   free-energy traces show why: the slow pre-trip precursor is real but
//!   tiny relative to calibration noise (for LOCA 1%, free energy reaches
//!   only ~7% of the alarm threshold by the time the trip fires at
//!   t=2032.5s); what actually crosses the threshold is the much larger
//!   whole-state deviation from the trip's *own* aftermath (control-rod
//!   insertion, flux collapse). This is a real limitation of
//!   equal-weighted whole-state cosine similarity — it drowns a slow,
//!   few-channel drift in channels that aren't moving yet — not a tuning
//!   bug worth chasing further here.
//! - The mildest severity of a slow, single-pathway accident (SG tube
//!   rupture at 1% of a 100-step severity scale) isn't detected at all
//!   within the ~4000s window, for the same underlying reason.
//! - The detector correctly stays quiet on the one ATWS fixture NPPAD
//!   ships — but that fixture turns out to encode "scram disabled, no
//!   other malfunction" rather than a runaway transient with failed
//!   scram (verified directly: PWR/PWNT settle to a flat 100.275% within
//!   ~1 minute and stay bit-identical for the rest of the window). That's
//!   correct specificity on a benign trace, not evidence the detector
//!   would catch a genuine escalating ATWS — NPPAD has no severity
//!   variants for this type to test that.
//! - Turbine trip (no scram ground truth to lag behind) is detected
//!   cleanly — the one case here that's a real, unconfounded win.
//!
//! Net: the whole-state detector alone is a **secondary confirmation
//! layer** that would echo a hardwired trip a few seconds after the fact,
//! not a generic early-warning system.
//!
//! ## Update (2026-07-07): [`PerChannelDetector`] closes part of that gap
//! Added to test the implied follow-up above: z-score each channel
//! against its own calibrated noise floor instead of blending all 21
//! into one HDC vector first. Measured result (see the
//! `test_per_channel_*` pinned tests):
//! - **LOCA 1% now beats the scram by ~32 minutes** (detected at t=130s via
//!   `PRB`, containment pressure, vs. the scram at t=2032.5s) — this *is*
//!   real early warning, not an echo, for a slow pressure-boundary leak.
//! - LOCA 50% and rod withdrawal still narrowly lag the scram (130s vs.
//!   127.5s; 430s vs. 403.5s) — a fast break moves containment pressure
//!   almost as fast as it moves the RCS pressure that trips the plant, and
//!   RW's precursor is a fast reactivity/flux excursion with little
//!   physical time between "diverging" and "crossing the flux setpoint."
//!   Different accident *physics*, not a shortfall in this detector.
//! - SGATR 1% and ATWS are still undetected — below the noise floor for
//!   every tracked channel at this severity, not just diluted by blending.
//!
//! So: whole-state detection is real value on slow degradation modes with
//! no existing single-threshold trip (per `fission_bench`'s own synthetic
//! fault library, where it *does* lead ground truth); per-channel
//! detection adds genuine early warning specifically for slow,
//! single-pathway accidents like a small LOCA, where the precursor is
//! large on one channel but diluted in a 21-channel blend. Neither
//! replaces a hardwired trip on fast, multi-channel events (LOCA 50%, RW)
//! — that's exactly the regime hardwired trips exist for.
//!
//! ## Other honesty notes
//! - Channel selection/normalization here is independent of
//!   `fission_bench`'s synthetic 21-channel design — different data
//!   domain, deliberately not sharing code with it (see module docs there).
//! - Only one healthy (`Normal`) run exists upstream; calibration and the
//!   false-alarm check both come from splitting that single continuous run.
//! - PCTRAN is a deterministic simulator — no instrument noise. A detector
//!   tuned here may need a wider margin against real sensor noise.
//! - This is advisory (non-1E) monitoring validation, not a claim about
//!   safety-grade reactor protection.

use std::collections::HashMap;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Channels extracted from the 97-column NPPAD schema, with the
/// normalization divisor (or transform) applied to each. Chosen to span
/// primary loop, secondary side, neutronics, containment, and
/// radiological — see `docs/TECHNICAL_REPORT`-style reasoning in the
/// fixtures README for why these seven cases were selected.
///
/// Ranges were checked against the actual fixture data (2026-07-07) so
/// these divisors don't saturate everything to 0 or 1 uselessly.
const CHANNELS: &[(&str, Normalizer)] = &[
    ("P", Normalizer::Linear(170.0)),     // RCS pressure (bar)
    ("TAVG", Normalizer::Linear(350.0)),  // RCS avg temp (°C)
    ("THA", Normalizer::Linear(350.0)),   // Hot leg A temp
    ("THB", Normalizer::Linear(350.0)),   // Hot leg B temp
    ("TCA", Normalizer::Linear(350.0)),   // Cold leg A temp
    ("TCB", Normalizer::Linear(350.0)),   // Cold leg B temp
    ("PSGA", Normalizer::Linear(90.0)),   // SG A pressure (bar)
    ("PSGB", Normalizer::Linear(90.0)),   // SG B pressure (bar)
    ("LVPZ", Normalizer::Linear(100.0)),  // Pressurizer level (%)
    ("WHPI", Normalizer::Linear(150.0)),  // HPI flow (t/hr)
    ("NSGA", Normalizer::Linear(100.0)),  // SG A level narrow range (%)
    ("NSGB", Normalizer::Linear(100.0)),  // SG B level narrow range (%)
    ("PRB", Normalizer::Linear(1.5)),     // Containment pressure (bar)
    ("TRB", Normalizer::Linear(100.0)),   // Containment temp (°C)
    ("DNBR", Normalizer::Capped(10.0)),   // Departure-from-boiling ratio
    ("PWNT", Normalizer::Linear(120.0)),  // Nuclear flux power (%)
    ("PWR", Normalizer::Linear(120.0)),   // Core thermal power (%)
    ("TPCT", Normalizer::Linear(700.0)),  // Peak clad temp (°C)
    ("RM1", Normalizer::Linear(1.0)),     // Containment air radiation
    ("RC131", Normalizer::Log(5.0, 2.0)), // Coolant I-131 activity (GBq/cc)
    ("PPM", Normalizer::Linear(1200.0)),  // Boron concentration (ppm)
];

pub const NPPAD_CHANNEL_COUNT: usize = 21;

/// How to map a raw column value into `[0, 1]`.
#[derive(Debug, Clone, Copy)]
enum Normalizer {
    /// `(value / scale).clamp(0, 1)`.
    Linear(f64),
    /// `(value.min(cap) / cap).clamp(0, 1)` — bounds an unbounded ratio
    /// (DNBR spikes to 10⁴ at near-zero power; only the low, dangerous
    /// end is informative).
    Capped(f64),
    /// `((log10(value) - floor) / span).clamp(0, 1)` — for quantities
    /// spanning multiple orders of magnitude (radionuclide activity).
    Log(f64, f64),
}

impl Normalizer {
    fn apply(&self, value: f64) -> f64 {
        let x = match *self {
            Normalizer::Linear(scale) => value / scale,
            Normalizer::Capped(cap) => value.min(cap) / cap,
            Normalizer::Log(floor, span) => {
                if value <= 0.0 {
                    0.0
                } else {
                    (value.log10() - floor) / span
                }
            }
        };
        x.clamp(0.0, 1.0)
    }
}

/// One parsed, normalized NPPAD data row.
#[derive(Debug, Clone)]
pub struct NppadRow {
    pub time_s: f64,
    pub normalized: [f64; NPPAD_CHANNEL_COUNT],
}

/// Parse an NPPAD CSV (header + data rows) into normalized channel
/// vectors. Panics if a required column is missing — the fixtures'
/// shared header is verified in [`tests::test_all_fixtures_same_schema`].
pub fn parse_nppad_csv(csv: &str) -> Vec<NppadRow> {
    let mut lines = csv.lines();
    let header = lines.next().expect("CSV must have a header line");
    let columns: Vec<&str> = header.split(',').collect();
    let index_of: HashMap<&str, usize> = columns.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let time_idx = *index_of.get("TIME").expect("missing TIME column");
    let channel_indices: Vec<usize> = CHANNELS
        .iter()
        .map(|(name, _)| {
            *index_of
                .get(name)
                .unwrap_or_else(|| panic!("missing expected NPPAD column: {name}"))
        })
        .collect();

    lines
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let fields: Vec<f64> = line.split(',').map(|v| v.parse().unwrap_or(0.0)).collect();
            let mut normalized = [0.0; NPPAD_CHANNEL_COUNT];
            for (i, &col_idx) in channel_indices.iter().enumerate() {
                normalized[i] = CHANNELS[i].1.apply(fields[col_idx]);
            }
            NppadRow {
                time_s: fields[time_idx],
                normalized,
            }
        })
        .collect()
}

/// HDC encoder for the reduced NPPAD channel set.
pub struct NppadEncoder {
    bases: [ContinuousHV; NPPAD_CHANNEL_COUNT],
}

impl NppadEncoder {
    pub fn new() -> Self {
        let bases =
            std::array::from_fn(|i| ContinuousHV::random(HDC_DIMENSION, 0xF15_3000 + i as u64));
        Self { bases }
    }

    pub fn encode(&self, normalized: &[f64; NPPAD_CHANNEL_COUNT]) -> ContinuousHV {
        let weights: [f32; NPPAD_CHANNEL_COUNT] = std::array::from_fn(|i| normalized[i] as f32);
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for NppadEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibrated free-energy detector over the NPPAD channel set. Same
/// calibrated-kσ/persistence approach as
/// [`crate::fission_bench::FreeEnergyDetector`], reimplemented rather
/// than shared: different channel count/domain, and this module must
/// stand as an independent check, not an extension of the synthetic one.
pub struct NppadDetector {
    encoder: NppadEncoder,
    reference: ContinuousHV,
    k_sigma: f64,
    persistence: usize,
    baseline_mean: f64,
    baseline_std: f64,
    consecutive: usize,
}

impl NppadDetector {
    pub fn new(
        reference_state: &[f64; NPPAD_CHANNEL_COUNT],
        k_sigma: f64,
        persistence: usize,
    ) -> Self {
        let encoder = NppadEncoder::new();
        let reference = encoder.encode(reference_state);
        Self {
            encoder,
            reference,
            k_sigma,
            persistence,
            baseline_mean: 0.0,
            baseline_std: 0.0,
            consecutive: 0,
        }
    }

    pub fn free_energy(&self, normalized: &[f64; NPPAD_CHANNEL_COUNT]) -> f64 {
        let hv = self.encoder.encode(normalized);
        let sim = hv.similarity(&self.reference) as f64;
        if !sim.is_finite() {
            1.0
        } else {
            (1.0 - sim).max(0.0)
        }
    }

    pub fn calibrate(&mut self, healthy: &[[f64; NPPAD_CHANNEL_COUNT]]) {
        assert!(healthy.len() >= 10, "need >=10 calibration samples");
        let fes: Vec<f64> = healthy.iter().map(|s| self.free_energy(s)).collect();
        let n = fes.len() as f64;
        let mean = fes.iter().sum::<f64>() / n;
        let var = fes.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
        self.baseline_mean = mean;
        self.baseline_std = var.sqrt().max(1e-9);
        self.consecutive = 0;
    }

    pub fn threshold(&self) -> f64 {
        self.baseline_mean + self.k_sigma * self.baseline_std
    }

    pub fn observe(&mut self, normalized: &[f64; NPPAD_CHANNEL_COUNT]) -> bool {
        if self.free_energy(normalized) > self.threshold() {
            self.consecutive += 1;
        } else {
            self.consecutive = 0;
        }
        self.consecutive >= self.persistence
    }
}

/// Per-channel name lookup, index-aligned with the normalized arrays
/// (avoids a second hand-maintained name list that could drift from
/// [`CHANNELS`]).
pub fn channel_name(index: usize) -> &'static str {
    CHANNELS[index].0
}

/// Per-channel calibrated z-score detector — complements [`NppadDetector`].
///
/// The whole-state detector blends all 21 channels into one HDC vector
/// before comparing to the reference; a drift confined to one or two
/// channels gets diluted by the ~19 that aren't moving yet (this is
/// exactly the mechanism documented in the module docs for why LOCA/RW
/// detection lags the scram, and why SGATR-1% isn't detected at all).
/// This detector instead z-scores each channel against *its own*
/// calibrated noise floor, undiluted by the others.
///
/// Testing 21 channels independently per sample is a multiple-comparisons
/// problem: a k_sigma tuned for a false-alarm rate on *one* channel gives
/// a much higher effective false-alarm rate across 21. `persistence`
/// (consecutive over-threshold samples on the *same* channel) is the
/// control for that — tune both together against
/// [`tests::test_per_channel_no_false_alarms_on_held_out_normal`], not in
/// isolation.
pub struct PerChannelDetector {
    k_sigma: f64,
    persistence: usize,
    baseline_mean: [f64; NPPAD_CHANNEL_COUNT],
    baseline_std: [f64; NPPAD_CHANNEL_COUNT],
    consecutive: [usize; NPPAD_CHANNEL_COUNT],
}

impl PerChannelDetector {
    pub fn new(k_sigma: f64, persistence: usize) -> Self {
        Self {
            k_sigma,
            persistence,
            baseline_mean: [0.0; NPPAD_CHANNEL_COUNT],
            baseline_std: [0.0; NPPAD_CHANNEL_COUNT],
            consecutive: [0; NPPAD_CHANNEL_COUNT],
        }
    }

    pub fn calibrate(&mut self, healthy: &[[f64; NPPAD_CHANNEL_COUNT]]) {
        assert!(healthy.len() >= 10, "need >=10 calibration samples");
        let n = healthy.len() as f64;
        for ch in 0..NPPAD_CHANNEL_COUNT {
            let mean = healthy.iter().map(|s| s[ch]).sum::<f64>() / n;
            let var = healthy.iter().map(|s| (s[ch] - mean).powi(2)).sum::<f64>() / n;
            self.baseline_mean[ch] = mean;
            self.baseline_std[ch] = var.sqrt().max(1e-9);
        }
        self.consecutive = [0; NPPAD_CHANNEL_COUNT];
    }

    /// Per-channel absolute z-score for one sample.
    pub fn z_scores(&self, normalized: &[f64; NPPAD_CHANNEL_COUNT]) -> [f64; NPPAD_CHANNEL_COUNT] {
        std::array::from_fn(|ch| {
            (normalized[ch] - self.baseline_mean[ch]).abs() / self.baseline_std[ch]
        })
    }

    /// Feed one sample. Returns the index of the first channel whose
    /// persistence counter crosses the alarm threshold this step, if any
    /// (an alarm can only newly fire on one channel per call, but other
    /// channels' counters still advance underneath).
    pub fn observe(&mut self, normalized: &[f64; NPPAD_CHANNEL_COUNT]) -> Option<usize> {
        let z = self.z_scores(normalized);
        let mut fired = None;
        for ch in 0..NPPAD_CHANNEL_COUNT {
            if z[ch] > self.k_sigma {
                self.consecutive[ch] += 1;
            } else {
                self.consecutive[ch] = 0;
            }
            if self.consecutive[ch] >= self.persistence && fired.is_none() {
                fired = Some(ch);
            }
        }
        fired
    }
}

/// Ground truth for one accident case (see fixtures `README.md`).
#[derive(Debug, Clone, Copy)]
pub struct NppadCase {
    pub name: &'static str,
    pub csv: &'static str,
    /// Accident/malfunction onset (s) — from upstream `*Transient
    /// Report.txt`; every case here has onset ≈ 0.5s.
    pub onset_s: f64,
    /// Automatic reactor scram time (s), if the plant's own protection
    /// fired within the fixture window. `None` means no scram occurred
    /// (either the transient didn't warrant one, or — for ATWS — the
    /// protection is defined not to act at all).
    pub scram_s: Option<f64>,
}

pub const NPPAD_CASES: &[NppadCase] = &[
    NppadCase {
        name: "LOCA (1% hot-leg break)",
        csv: include_str!("../tests/fixtures/nppad/LOCA_1.csv"),
        onset_s: 0.5,
        scram_s: Some(2032.5),
    },
    NppadCase {
        name: "LOCA (50% hot-leg break)",
        csv: include_str!("../tests/fixtures/nppad/LOCA_50.csv"),
        onset_s: 0.5,
        scram_s: Some(127.5),
    },
    NppadCase {
        name: "Rod withdrawal (1%)",
        csv: include_str!("../tests/fixtures/nppad/RW_1.csv"),
        onset_s: 0.5,
        scram_s: Some(403.5),
    },
    NppadCase {
        name: "Turbine trip",
        csv: include_str!("../tests/fixtures/nppad/TT_1.csv"),
        onset_s: 0.5,
        scram_s: None,
    },
    NppadCase {
        name: "SG-A tube rupture (1%)",
        csv: include_str!("../tests/fixtures/nppad/SGATR_1.csv"),
        onset_s: 0.5,
        scram_s: None,
    },
    NppadCase {
        name: "ATWS",
        csv: include_str!("../tests/fixtures/nppad/ATWS_1.csv"),
        onset_s: 0.5,
        scram_s: None, // by construction: this is the no-scram case.
    },
];

pub const NPPAD_NORMAL_CSV: &str = include_str!("../tests/fixtures/nppad/Normal_1.csv");

/// Recommended detector configuration for the NPPAD validation (distinct
/// from `fission_bench`'s constants — see module docs).
pub const NPPAD_K_SIGMA: f64 = 3.0;
pub const NPPAD_PERSISTENCE: usize = 1;

/// Outcome of running one [`NppadCase`] against a freshly calibrated
/// detector.
#[derive(Debug, Clone)]
pub struct NppadResult {
    pub case_name: &'static str,
    pub detected: bool,
    pub detection_time_s: Option<f64>,
    pub detection_latency_s: Option<f64>,
    /// Whether detection beat the automatic scram, when one occurred.
    pub beat_scram: Option<bool>,
}

/// Calibrate on the first `calib_rows` of the Normal run, then run every
/// [`NPPAD_CASES`] entry against a fresh detector instance.
pub fn run_all_cases(calib_rows: usize) -> Vec<NppadResult> {
    let normal = parse_nppad_csv(NPPAD_NORMAL_CSV);
    let calib: Vec<[f64; NPPAD_CHANNEL_COUNT]> =
        normal[..calib_rows].iter().map(|r| r.normalized).collect();
    let reference = normal[0].normalized;

    NPPAD_CASES
        .iter()
        .map(|case| {
            let mut det = NppadDetector::new(&reference, NPPAD_K_SIGMA, NPPAD_PERSISTENCE);
            det.calibrate(&calib);
            let rows = parse_nppad_csv(case.csv);
            let mut detection_time_s = None;
            for row in &rows {
                if det.observe(&row.normalized) {
                    detection_time_s = Some(row.time_s);
                    break;
                }
            }
            let detection_latency_s = detection_time_s.map(|t| t - case.onset_s);
            let beat_scram = match (detection_time_s, case.scram_s) {
                (Some(dt), Some(scram)) => Some(dt < scram),
                _ => None,
            };
            NppadResult {
                case_name: case.name,
                detected: detection_time_s.is_some(),
                detection_time_s,
                detection_latency_s,
                beat_scram,
            }
        })
        .collect()
}

/// Recommended per-channel detector configuration — tuned empirically
/// against [`tests::test_per_channel_no_false_alarms_on_held_out_normal`]
/// (higher than [`NPPAD_K_SIGMA`]/[`NPPAD_PERSISTENCE`] precisely because
/// of the multiple-comparisons problem across 21 independently-tested
/// channels — see [`PerChannelDetector`] docs).
pub const PER_CHANNEL_K_SIGMA: f64 = 5.0;
pub const PER_CHANNEL_PERSISTENCE: usize = 3;

/// Outcome of running one [`NppadCase`] against a freshly calibrated
/// [`PerChannelDetector`].
#[derive(Debug, Clone)]
pub struct NppadPerChannelResult {
    pub case_name: &'static str,
    pub detected: bool,
    pub detection_time_s: Option<f64>,
    pub detection_latency_s: Option<f64>,
    /// Which channel's persistence counter first crossed threshold.
    pub detection_channel: Option<&'static str>,
    pub beat_scram: Option<bool>,
}

/// Same protocol as [`run_all_cases`], but with [`PerChannelDetector`]
/// instead of the whole-state [`NppadDetector`] — lets the two approaches
/// be compared directly against the same fixtures and ground truth.
pub fn run_all_cases_per_channel(calib_rows: usize) -> Vec<NppadPerChannelResult> {
    let normal = parse_nppad_csv(NPPAD_NORMAL_CSV);
    let calib: Vec<[f64; NPPAD_CHANNEL_COUNT]> =
        normal[..calib_rows].iter().map(|r| r.normalized).collect();

    NPPAD_CASES
        .iter()
        .map(|case| {
            let mut det = PerChannelDetector::new(PER_CHANNEL_K_SIGMA, PER_CHANNEL_PERSISTENCE);
            det.calibrate(&calib);
            let rows = parse_nppad_csv(case.csv);
            let mut detection_time_s = None;
            let mut detection_channel = None;
            for row in &rows {
                if let Some(ch) = det.observe(&row.normalized) {
                    detection_time_s = Some(row.time_s);
                    detection_channel = Some(channel_name(ch));
                    break;
                }
            }
            let detection_latency_s = detection_time_s.map(|t| t - case.onset_s);
            let beat_scram = match (detection_time_s, case.scram_s) {
                (Some(dt), Some(scram)) => Some(dt < scram),
                _ => None,
            };
            NppadPerChannelResult {
                case_name: case.name,
                detected: detection_time_s.is_some(),
                detection_time_s,
                detection_latency_s,
                detection_channel,
                beat_scram,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_fixtures_same_schema() {
        let normal_header = NPPAD_NORMAL_CSV.lines().next().unwrap();
        for case in NPPAD_CASES {
            assert_eq!(
                case.csv.lines().next().unwrap(),
                normal_header,
                "{} header mismatch",
                case.name
            );
        }
    }

    #[test]
    fn test_parse_normal_in_bounds() {
        let rows = parse_nppad_csv(NPPAD_NORMAL_CSV);
        assert!(
            rows.len() > 200,
            "expected a substantial Normal run, got {}",
            rows.len()
        );
        for row in &rows {
            for (i, &x) in row.normalized.iter().enumerate() {
                assert!((0.0..=1.0).contains(&x), "channel {i} out of bounds: {x}");
            }
        }
    }

    #[test]
    fn test_parse_time_monotonic() {
        let rows = parse_nppad_csv(NPPAD_NORMAL_CSV);
        for w in rows.windows(2) {
            assert!(w[1].time_s > w[0].time_s);
        }
    }

    #[test]
    fn test_all_cases_parse_and_start_near_onset() {
        for case in NPPAD_CASES {
            let rows = parse_nppad_csv(case.csv);
            assert!(!rows.is_empty(), "{} produced no rows", case.name);
            assert!(
                rows[0].time_s <= case.onset_s + 1.0,
                "{} doesn't start at onset",
                case.name
            );
        }
    }

    #[test]
    fn test_calibration_threshold_positive() {
        let normal = parse_nppad_csv(NPPAD_NORMAL_CSV);
        let calib: Vec<_> = normal[..200].iter().map(|r| r.normalized).collect();
        let mut det = NppadDetector::new(&normal[0].normalized, NPPAD_K_SIGMA, NPPAD_PERSISTENCE);
        det.calibrate(&calib);
        assert!(det.threshold() >= 0.0 && det.threshold().is_finite());
    }

    #[test]
    fn test_no_false_alarms_on_held_out_normal() {
        let normal = parse_nppad_csv(NPPAD_NORMAL_CSV);
        let split = 200.min(normal.len() - 20);
        let calib: Vec<_> = normal[..split].iter().map(|r| r.normalized).collect();
        let mut det = NppadDetector::new(&normal[0].normalized, NPPAD_K_SIGMA, NPPAD_PERSISTENCE);
        det.calibrate(&calib);
        let mut alarms = 0;
        for row in &normal[split..] {
            if det.observe(&row.normalized) {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0, "false alarm(s) on held-out healthy data");
    }

    fn scram_s_of(name: &str) -> f64 {
        NPPAD_CASES
            .iter()
            .find(|c| c.name == name)
            .and_then(|c| c.scram_s)
            .expect("case must have a known scram time")
    }

    // The four tests below pin the ACTUAL measured behavior of the
    // calibrated whole-state HDC detector against real PCTRAN data, not
    // an aspirational one. Diagnostic free-energy traces (2026-07-07)
    // showed why: for LOCA/RW, the pre-trip precursor is a slow, small,
    // single/few-channel drift that stays far below any sane kσ threshold
    // for the whole run — the detector's alarm is actually driven by the
    // automatic scram's OWN aftermath (control-rod insertion + flux
    // collapse), which is a much larger whole-state deviation than the
    // accident precursor itself. So the detector reliably fires shortly
    // AFTER the plant's own hardwired trip, never before, regardless of
    // reasonable kσ/persistence tuning (tried 6σ/persist-2 down to
    // 3σ/persist-1; the LOCA-1 pre-trip free energy only reaches ~7% of
    // threshold by the time the trip fires). This is a genuine limitation
    // of equal-weighted whole-state cosine similarity, not a tuning bug —
    // see module docs "Honesty notes" for what it implies.
    #[test]
    fn test_loca_1_detected_shortly_after_scram() {
        let r = run_all_cases(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("LOCA (1%"))
            .unwrap();
        assert!(r.detected, "LOCA 1% not detected at all: {r:?}");
        assert_eq!(
            r.beat_scram,
            Some(false),
            "expected the known lag-after-scram behavior: {r:?}"
        );
        let lag = r.detection_time_s.unwrap() - scram_s_of(r.case_name);
        assert!(
            (0.0..60.0).contains(&lag),
            "lag outside expected 0-60s band: {lag}, {r:?}"
        );
    }

    #[test]
    fn test_loca_50_detected_shortly_after_scram() {
        let r = run_all_cases(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("LOCA (50%"))
            .unwrap();
        assert!(r.detected, "LOCA 50% not detected at all: {r:?}");
        assert_eq!(
            r.beat_scram,
            Some(false),
            "expected the known lag-after-scram behavior: {r:?}"
        );
        let lag = r.detection_time_s.unwrap() - scram_s_of(r.case_name);
        assert!(
            (0.0..60.0).contains(&lag),
            "lag outside expected 0-60s band: {lag}, {r:?}"
        );
    }

    #[test]
    fn test_rod_withdrawal_detected_shortly_after_scram() {
        let r = run_all_cases(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("Rod withdrawal"))
            .unwrap();
        assert!(r.detected, "rod withdrawal not detected at all: {r:?}");
        assert_eq!(
            r.beat_scram,
            Some(false),
            "expected the known lag-after-scram behavior: {r:?}"
        );
        let lag = r.detection_time_s.unwrap() - scram_s_of(r.case_name);
        assert!(
            (0.0..60.0).contains(&lag),
            "lag outside expected 0-60s band: {lag}, {r:?}"
        );
    }

    #[test]
    fn test_turbine_trip_detected() {
        // No automatic scram occurs for this case (relief valves handle
        // it), so there's no lag-after-trip confound — this is a genuine
        // detection of the turbine-trip transient itself.
        let r = run_all_cases(200)
            .into_iter()
            .find(|r| r.case_name == "Turbine trip")
            .unwrap();
        assert!(r.detected, "turbine trip not detected: {r:?}");
    }

    /// Pinned finding, not a bug: the mildest severity of this accident
    /// type (1% of a full tube rupture — the gentlest end of a 100-step
    /// severity scale) does not cross the whole-state threshold within
    /// the ~4000s fixture window. A slow, mild, single-pathway
    /// (radiological) leak is exactly the kind of signal equal-weighted
    /// whole-state cosine similarity under-detects — see module docs.
    #[test]
    fn test_sgatr_1pct_not_detected_within_window() {
        let r = run_all_cases(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("SG-A"))
            .unwrap();
        assert!(
            !r.detected,
            "SGATR 1% is now detected — update this pinned finding: {r:?}"
        );
    }

    /// Pinned finding, not a bug: per the upstream `*Transient
    /// Report.txt`, this specific ATWS fixture logs only "scram capability
    /// disabled" (Malfunction #8, fraction 0%) with no paired initiating
    /// event. Checked directly (2026-07-07): PWR/PWNT settle from 100% to
    /// a flat 100.275% within ~1 minute and stay bit-identical for the
    /// rest of the 4000s window; RH (total reactivity) sits at ~0. This
    /// fixture is the *benign* end of ATWS — protection disabled, but
    /// nothing else goes wrong — not a runaway transient with failed
    /// scram. Correctly staying quiet here is a specificity finding, not
    /// a missed detection. NPPAD ships no other ATWS severity to test the
    /// escalating case.
    #[test]
    fn test_atws_correctly_quiet_on_benign_fixture() {
        let r = run_all_cases(200)
            .into_iter()
            .find(|r| r.case_name == "ATWS")
            .unwrap();
        assert!(
            !r.detected,
            "ATWS fixture now shows a signal — investigate before updating: {r:?}"
        );
    }

    #[test]
    fn test_all_cases_run_without_panic() {
        let results = run_all_cases(200);
        assert_eq!(results.len(), NPPAD_CASES.len());
    }

    // --- PerChannelDetector: the follow-up implied in the module docs ---
    //
    // Measured 2026-07-07 at PER_CHANNEL_K_SIGMA=5.0 / PER_CHANNEL_PERSISTENCE=3
    // (tuned against test_per_channel_no_false_alarms_on_held_out_normal
    // below — 0 false alarms). Headline result: the slow LOCA (1%) precursor
    // that the whole-state detector could only echo ~7.5s after the scram
    // is, on its own channel (PRB, containment pressure), large enough to
    // alarm at t=130s — nearly 32 minutes *before* the scram at t=2032.5s.
    // This is the real early-warning result the whole-state detector
    // structurally cannot give (see its module-doc "Result" section).
    //
    // Not a universal win, and the tests below pin the honest boundary:
    // - LOCA 50% (fast break): per-channel still narrowly loses (130s vs
    //   scram at 127.5s) — a dramatic break moves containment pressure
    //   almost as fast as it moves the RCS pressure that triggers the trip.
    // - Rod withdrawal: per-channel still loses (430s vs 403.5s) — RW's
    //   precursor is a fast reactivity/flux excursion, not a slow
    //   pressure-boundary leak; there's little physical time between
    //   "diverging" and "crossing the flux scram setpoint" for this event
    //   type. Different accident *physics*, not a detector shortcoming.
    // - SGATR 1% and ATWS: still undetected, same as the whole-state
    //   detector — the signal is below the noise floor for every tracked
    //   channel at this severity, not just diluted by blending.

    #[test]
    fn test_per_channel_no_false_alarms_on_held_out_normal() {
        let normal = parse_nppad_csv(NPPAD_NORMAL_CSV);
        let split = 200.min(normal.len() - 20);
        let calib: Vec<_> = normal[..split].iter().map(|r| r.normalized).collect();
        let mut det = PerChannelDetector::new(PER_CHANNEL_K_SIGMA, PER_CHANNEL_PERSISTENCE);
        det.calibrate(&calib);
        let mut alarms = 0;
        for row in &normal[split..] {
            if det.observe(&row.normalized).is_some() {
                alarms += 1;
            }
        }
        assert_eq!(alarms, 0, "false alarm(s) on held-out healthy data");
    }

    #[test]
    fn test_per_channel_loca_1_beats_scram() {
        let r = run_all_cases_per_channel(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("LOCA (1%"))
            .unwrap();
        assert!(r.detected, "LOCA 1% not detected: {r:?}");
        assert_eq!(
            r.beat_scram,
            Some(true),
            "expected per-channel to beat the scram here: {r:?}"
        );
        assert_eq!(
            r.detection_channel,
            Some("PRB"),
            "detecting channel changed: {r:?}"
        );
    }

    #[test]
    fn test_per_channel_loca_50_still_narrowly_lags_scram() {
        let r = run_all_cases_per_channel(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("LOCA (50%"))
            .unwrap();
        assert!(r.detected, "LOCA 50% not detected: {r:?}");
        assert_eq!(
            r.beat_scram,
            Some(false),
            "a large break moves containment pressure almost as fast as RCS \
             pressure — if this now beats scram, that's a real improvement, \
             update this pinned finding: {r:?}"
        );
    }

    #[test]
    fn test_per_channel_rod_withdrawal_still_lags_scram() {
        let r = run_all_cases_per_channel(200)
            .into_iter()
            .find(|r| r.case_name.starts_with("Rod withdrawal"))
            .unwrap();
        assert!(r.detected, "rod withdrawal not detected: {r:?}");
        assert_eq!(
            r.beat_scram,
            Some(false),
            "RW's precursor is a fast reactivity excursion, not a slow leak — \
             if this now beats scram, update this pinned finding: {r:?}"
        );
    }

    #[test]
    fn test_per_channel_turbine_trip_detected() {
        let r = run_all_cases_per_channel(200)
            .into_iter()
            .find(|r| r.case_name == "Turbine trip")
            .unwrap();
        assert!(r.detected, "turbine trip not detected: {r:?}");
    }

    #[test]
    fn test_per_channel_sgatr_and_atws_still_undetected() {
        let results = run_all_cases_per_channel(200);
        for name in ["SG-A tube rupture (1%)", "ATWS"] {
            let r = results.iter().find(|r| r.case_name == name).unwrap();
            assert!(
                !r.detected,
                "{name} is now detected by the per-channel detector — the \
                 signal crossed a channel's noise floor where it didn't \
                 before; update this pinned finding: {r:?}"
            );
        }
    }
}
