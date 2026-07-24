// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Target-independent quality metrics for normalized articulatory trajectories.

use serde::{Deserialize, Serialize};

use crate::GestureFrame;

pub const ARTICULATORY_QUALITY_SCHEMA: &str = "symthaea.articulatory-quality.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulatoryQualityMetrics {
    pub schema: String,
    pub frames_evaluated: usize,
    pub voiced_frames: usize,
    pub silent_frames: usize,
    pub maximum_coordinate_slew_per_second: f32,
    pub maximum_f0_octaves_per_second: f32,
    pub maximum_energy_slew_per_second: f32,
    pub silence_leakage_frames: usize,
    pub non_finite_measurements: usize,
}

impl ArticulatoryQualityMetrics {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema != ARTICULATORY_QUALITY_SCHEMA
            || self.voiced_frames > self.frames_evaluated
            || self.silent_frames > self.frames_evaluated
            || self.silence_leakage_frames > self.silent_frames
            || !self.maximum_coordinate_slew_per_second.is_finite()
            || !self.maximum_f0_octaves_per_second.is_finite()
            || !self.maximum_energy_slew_per_second.is_finite()
        {
            return Err("invalid articulatory quality metrics");
        }
        Ok(())
    }

    pub fn gate(
        &self,
        requirements: &ArticulatoryQualityRequirements,
    ) -> Result<ArticulatoryQualityGate, &'static str> {
        self.validate()?;
        requirements.validate()?;
        let coverage_pass = self.frames_evaluated >= requirements.minimum_frames;
        let finite_pass = self.non_finite_measurements == 0;
        let coordinate_slew_pass = coverage_pass
            && self.maximum_coordinate_slew_per_second
                <= requirements.maximum_coordinate_slew_per_second;
        let f0_slew_pass = coverage_pass
            && self.maximum_f0_octaves_per_second
                <= requirements.maximum_f0_octaves_per_second;
        let energy_slew_pass = coverage_pass
            && self.maximum_energy_slew_per_second
                <= requirements.maximum_energy_slew_per_second;
        let silence_pass = !requirements.require_silence_coverage
            || (self.silent_frames > 0 && self.silence_leakage_frames == 0);
        Ok(ArticulatoryQualityGate {
            coverage_pass,
            finite_pass,
            coordinate_slew_pass,
            f0_slew_pass,
            energy_slew_pass,
            silence_pass,
            pass: coverage_pass
                && finite_pass
                && coordinate_slew_pass
                && f0_slew_pass
                && energy_slew_pass
                && silence_pass,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ArticulatoryQualityRequirements {
    pub minimum_frames: usize,
    pub maximum_coordinate_slew_per_second: f32,
    pub maximum_f0_octaves_per_second: f32,
    pub maximum_energy_slew_per_second: f32,
    pub require_silence_coverage: bool,
}

impl Default for ArticulatoryQualityRequirements {
    fn default() -> Self {
        Self {
            minimum_frames: 8,
            maximum_coordinate_slew_per_second: 80.0,
            maximum_f0_octaves_per_second: 10.0,
            maximum_energy_slew_per_second: 80.0,
            require_silence_coverage: false,
        }
    }
}

impl ArticulatoryQualityRequirements {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.minimum_frames < 2
            || !self.maximum_coordinate_slew_per_second.is_finite()
            || self.maximum_coordinate_slew_per_second <= 0.0
            || !self.maximum_f0_octaves_per_second.is_finite()
            || self.maximum_f0_octaves_per_second <= 0.0
            || !self.maximum_energy_slew_per_second.is_finite()
            || self.maximum_energy_slew_per_second <= 0.0
        {
            return Err("invalid articulatory quality requirements");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArticulatoryQualityGate {
    pub coverage_pass: bool,
    pub finite_pass: bool,
    pub coordinate_slew_pass: bool,
    pub f0_slew_pass: bool,
    pub energy_slew_pass: bool,
    pub silence_pass: bool,
    pub pass: bool,
}

pub fn analyze_articulatory_trajectory(
    frames: &[GestureFrame],
    frame_rate_hz: f32,
) -> Result<ArticulatoryQualityMetrics, &'static str> {
    if frames.is_empty() || !frame_rate_hz.is_finite() || frame_rate_hz <= 0.0 {
        return Err("articulatory analysis requires frames and a valid clock");
    }
    let mut maximum_coordinate_slew = 0.0f32;
    let mut maximum_f0_slew = 0.0f32;
    let mut maximum_energy_slew = 0.0f32;
    let mut non_finite = 0usize;

    for frame in frames {
        if frame.validate().is_err() {
            non_finite = non_finite.saturating_add(1);
        }
    }
    for pair in frames.windows(2) {
        for (left, right) in continuous_coordinates(&pair[0])
            .into_iter()
            .zip(continuous_coordinates(&pair[1]))
        {
            let slew = (right - left).abs() * frame_rate_hz;
            if slew.is_finite() {
                maximum_coordinate_slew = maximum_coordinate_slew.max(slew);
            } else {
                non_finite = non_finite.saturating_add(1);
            }
        }
        if is_voiced(&pair[0]) && is_voiced(&pair[1]) {
            let f0_slew = (pair[1].f0_hz / pair[0].f0_hz.max(20.0))
                .log2()
                .abs()
                * frame_rate_hz;
            if f0_slew.is_finite() {
                maximum_f0_slew = maximum_f0_slew.max(f0_slew);
            } else {
                non_finite = non_finite.saturating_add(1);
            }
        }
        let energy_slew = (pair[1].energy.get() - pair[0].energy.get()).abs() * frame_rate_hz;
        if energy_slew.is_finite() {
            maximum_energy_slew = maximum_energy_slew.max(energy_slew);
        } else {
            non_finite = non_finite.saturating_add(1);
        }
    }

    let silent_frames = frames
        .iter()
        .filter(|frame| frame.energy.get() <= 1e-6)
        .count();
    let silence_leakage_frames = frames
        .iter()
        .filter(|frame| {
            frame.energy.get() <= 1e-6
                && (frame.glottal_adduction.get() > 1e-6
                    || frame.respiratory_effort.get() > 1e-6)
        })
        .count();
    let metrics = ArticulatoryQualityMetrics {
        schema: ARTICULATORY_QUALITY_SCHEMA.to_owned(),
        frames_evaluated: frames.len(),
        voiced_frames: frames.iter().filter(|frame| is_voiced(frame)).count(),
        silent_frames,
        maximum_coordinate_slew_per_second: maximum_coordinate_slew,
        maximum_f0_octaves_per_second: maximum_f0_slew,
        maximum_energy_slew_per_second: maximum_energy_slew,
        silence_leakage_frames,
        non_finite_measurements: non_finite,
    };
    metrics.validate()?;
    Ok(metrics)
}


fn is_voiced(frame: &GestureFrame) -> bool {
    frame.glottal_adduction.get() >= 0.35 && frame.energy.get() > 1e-6
}

fn continuous_coordinates(frame: &GestureFrame) -> [f32; 13] {
    [
        frame.jaw_aperture.get(),
        frame.tongue_body_height.get(),
        frame.tongue_body_frontness.get(),
        frame.tongue_tip_constriction.get(),
        frame.tongue_tip_location.get(),
        frame.lip_aperture.get(),
        frame.lip_protrusion.get(),
        frame.velum_opening.get(),
        frame.pharyngeal_constriction.get(),
        frame.larynx_height.get(),
        frame.glottal_adduction.get(),
        frame.vocal_fold_tension.get(),
        frame.respiratory_effort.get(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArticulatoryGestureScheduler, ArticulatoryScore, TimedPhoneme, UnitInterval};

    fn event(symbol: &str, onset_s: f32, duration_s: f32) -> TimedPhoneme {
        TimedPhoneme {
            symbol: symbol.to_owned(),
            onset_s,
            duration_s,
            f0_start_hz: 180.0,
            f0_end_hz: 220.0,
            energy: UnitInterval::new(0.7),
        }
    }

    #[test]
    fn scheduler_trajectory_is_finite_and_silence_clean() {
        let score = ArticulatoryScore::new(vec![
            event("AY", 0.0, 0.16),
            event("SIL", 0.16, 0.08),
            event("N", 0.24, 0.08),
        ])
        .unwrap();
        let scheduler = ArticulatoryGestureScheduler::default();
        let frames = scheduler.render(&score).unwrap();
        let metrics = analyze_articulatory_trajectory(
            &frames,
            scheduler.config().frame_rate_hz,
        )
        .unwrap();
        assert_eq!(metrics.non_finite_measurements, 0);
        assert!(metrics.silent_frames > 0);
        assert_eq!(metrics.silence_leakage_frames, 0);
    }

    #[test]
    fn unvoiced_f0_placeholders_do_not_create_false_slew_failures() {
        let mut voiced = GestureFrame::default();
        voiced.f0_hz = 440.0;
        let mut silent = GestureFrame::default();
        silent.f0_hz = 20.0;
        silent.energy = UnitInterval::new(0.0);
        silent.glottal_adduction = UnitInterval::new(0.0);
        silent.respiratory_effort = UnitInterval::new(0.0);
        let metrics = analyze_articulatory_trajectory(&[voiced, silent], 200.0).unwrap();
        assert_eq!(metrics.maximum_f0_octaves_per_second, 0.0);
        assert_eq!(metrics.non_finite_measurements, 0);
    }

    #[test]
    fn silence_requirement_is_fail_closed_when_not_exercised() {
        let frames = vec![GestureFrame::default(); 12];
        let metrics = analyze_articulatory_trajectory(&frames, 200.0).unwrap();
        let requirements = ArticulatoryQualityRequirements {
            require_silence_coverage: true,
            ..ArticulatoryQualityRequirements::default()
        };
        let gate = metrics.gate(&requirements).unwrap();
        assert!(!gate.silence_pass);
        assert!(!gate.pass);
    }
}
