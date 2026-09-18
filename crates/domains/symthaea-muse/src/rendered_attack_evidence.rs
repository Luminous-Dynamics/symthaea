// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive matched-window audio evidence around one performed attack.
//!
//! This module deliberately does not decide whether a rendered event is an
//! acoustic onset, perceptually salient, audible, or musically preferable.
//! It records finite signal measurements for predeclared windows so a later,
//! independently calibrated gate can make those stronger claims.

use serde::{Deserialize, Serialize};

pub const RENDERED_ATTACK_EVIDENCE_VERSION: &str = "rendered-attack-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StereoWindowMetricsV1 {
    pub rms: f64,
    pub peak_absolute: f32,
    /// Mean absolute first difference, pooled over left/right channels.
    /// This is a time-domain activity descriptor, not spectral flux.
    pub mean_absolute_first_difference: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedAttackEvidenceV1 {
    pub evidence_version: String,
    pub sample_rate: u32,
    pub attack_time_seconds: f32,
    pub center_sample: usize,
    pub pre_window_seconds: f32,
    pub post_window_seconds: f32,
    pub pre_samples: usize,
    pub post_samples: usize,
    pub baseline_pre: StereoWindowMetricsV1,
    pub baseline_post: StereoWindowMetricsV1,
    pub candidate_pre: StereoWindowMetricsV1,
    pub candidate_post: StereoWindowMetricsV1,
    pub difference_pre: StereoWindowMetricsV1,
    pub difference_post: StereoWindowMetricsV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedAttackEvidenceErrorV1 {
    InvalidSampleRate,
    InvalidAttackTime,
    InvalidWindow,
    EmptyAudio,
    LengthMismatch,
    WindowOutOfBounds,
    NonFiniteAudio,
}

/// Measure matched baseline/candidate stereo windows around one performed
/// attack time. All geometry is predeclared by the caller; this function does
/// not search for a favorable peak or optimize a threshold after observing the
/// signal.
pub fn measure_rendered_attack(
    baseline: &[[f32; 2]],
    candidate: &[[f32; 2]],
    sample_rate: u32,
    attack_time_seconds: f32,
    pre_window_seconds: f32,
    post_window_seconds: f32,
) -> Result<RenderedAttackEvidenceV1, RenderedAttackEvidenceErrorV1> {
    if sample_rate == 0 {
        return Err(RenderedAttackEvidenceErrorV1::InvalidSampleRate);
    }
    if !attack_time_seconds.is_finite() || attack_time_seconds < 0.0 {
        return Err(RenderedAttackEvidenceErrorV1::InvalidAttackTime);
    }
    if !pre_window_seconds.is_finite()
        || !post_window_seconds.is_finite()
        || pre_window_seconds <= 0.0
        || post_window_seconds <= 0.0
    {
        return Err(RenderedAttackEvidenceErrorV1::InvalidWindow);
    }
    if baseline.is_empty() || candidate.is_empty() {
        return Err(RenderedAttackEvidenceErrorV1::EmptyAudio);
    }
    if baseline.len() != candidate.len() {
        return Err(RenderedAttackEvidenceErrorV1::LengthMismatch);
    }
    if !finite_audio(baseline) || !finite_audio(candidate) {
        return Err(RenderedAttackEvidenceErrorV1::NonFiniteAudio);
    }

    let rate = sample_rate as f32;
    let center_sample = (attack_time_seconds * rate).round() as usize;
    let pre_samples = (pre_window_seconds * rate).round().max(1.0) as usize;
    let post_samples = (post_window_seconds * rate).round().max(1.0) as usize;
    let Some(pre_start) = center_sample.checked_sub(pre_samples) else {
        return Err(RenderedAttackEvidenceErrorV1::WindowOutOfBounds);
    };
    let Some(post_end) = center_sample.checked_add(post_samples) else {
        return Err(RenderedAttackEvidenceErrorV1::WindowOutOfBounds);
    };
    if center_sample > baseline.len() || post_end > baseline.len() {
        return Err(RenderedAttackEvidenceErrorV1::WindowOutOfBounds);
    }

    let baseline_pre_frames = &baseline[pre_start..center_sample];
    let baseline_post_frames = &baseline[center_sample..post_end];
    let candidate_pre_frames = &candidate[pre_start..center_sample];
    let candidate_post_frames = &candidate[center_sample..post_end];

    Ok(RenderedAttackEvidenceV1 {
        evidence_version: RENDERED_ATTACK_EVIDENCE_VERSION.into(),
        sample_rate,
        attack_time_seconds,
        center_sample,
        pre_window_seconds,
        post_window_seconds,
        pre_samples,
        post_samples,
        baseline_pre: metrics(baseline_pre_frames),
        baseline_post: metrics(baseline_post_frames),
        candidate_pre: metrics(candidate_pre_frames),
        candidate_post: metrics(candidate_post_frames),
        difference_pre: difference_metrics(baseline_pre_frames, candidate_pre_frames),
        difference_post: difference_metrics(baseline_post_frames, candidate_post_frames),
    })
}

fn finite_audio(frames: &[[f32; 2]]) -> bool {
    frames
        .iter()
        .all(|frame| frame[0].is_finite() && frame[1].is_finite())
}

fn metrics(frames: &[[f32; 2]]) -> StereoWindowMetricsV1 {
    let mut squared_sum = 0.0_f64;
    let mut peak_absolute = 0.0_f32;
    for frame in frames {
        for sample in *frame {
            squared_sum += (sample as f64) * (sample as f64);
            peak_absolute = peak_absolute.max(sample.abs());
        }
    }

    let mut first_difference_sum = 0.0_f64;
    let mut first_difference_count = 0usize;
    for pair in frames.windows(2) {
        for channel in 0..2 {
            first_difference_sum += (pair[1][channel] - pair[0][channel]).abs() as f64;
            first_difference_count += 1;
        }
    }

    StereoWindowMetricsV1 {
        rms: (squared_sum / (frames.len() * 2) as f64).sqrt(),
        peak_absolute,
        mean_absolute_first_difference: if first_difference_count == 0 {
            0.0
        } else {
            first_difference_sum / first_difference_count as f64
        },
    }
}

fn difference_metrics(left: &[[f32; 2]], right: &[[f32; 2]]) -> StereoWindowMetricsV1 {
    debug_assert_eq!(left.len(), right.len());
    let difference: Vec<[f32; 2]> = left
        .iter()
        .zip(right)
        .map(|(left, right)| [right[0] - left[0], right[1] - left[1]])
        .collect();
    metrics(&difference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_audio_retains_zero_difference_windows() {
        let audio = vec![[0.25, -0.25]; 1_000];
        let evidence = measure_rendered_attack(&audio, &audio, 1_000, 0.5, 0.1, 0.1).unwrap();
        assert_eq!(evidence.center_sample, 500);
        assert_eq!(evidence.pre_samples, 100);
        assert_eq!(evidence.post_samples, 100);
        assert_eq!(evidence.difference_pre.rms, 0.0);
        assert_eq!(evidence.difference_post.rms, 0.0);
        assert_eq!(evidence.difference_post.peak_absolute, 0.0);
    }

    #[test]
    fn synthetic_post_attack_change_is_retained_without_a_verdict() {
        let baseline = vec![[0.0, 0.0]; 1_000];
        let mut candidate = baseline.clone();
        for frame in &mut candidate[500..550] {
            *frame = [0.5, -0.25];
        }
        let evidence =
            measure_rendered_attack(&baseline, &candidate, 1_000, 0.5, 0.1, 0.1).unwrap();
        assert_eq!(evidence.difference_pre.rms, 0.0);
        assert!(evidence.difference_post.rms > 0.0);
        assert_eq!(evidence.difference_post.peak_absolute, 0.5);
        assert!(evidence.candidate_post.mean_absolute_first_difference >= 0.0);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let audio = vec![[0.0, 0.0]; 100];
        assert_eq!(
            measure_rendered_attack(&audio, &audio, 0, 0.05, 0.01, 0.01),
            Err(RenderedAttackEvidenceErrorV1::InvalidSampleRate)
        );
        assert_eq!(
            measure_rendered_attack(&audio, &audio, 1_000, f32::NAN, 0.01, 0.01),
            Err(RenderedAttackEvidenceErrorV1::InvalidAttackTime)
        );
        assert_eq!(
            measure_rendered_attack(&audio, &audio[..99], 1_000, 0.05, 0.01, 0.01),
            Err(RenderedAttackEvidenceErrorV1::LengthMismatch)
        );

        let mut non_finite = audio.clone();
        non_finite[50][0] = f32::INFINITY;
        assert_eq!(
            measure_rendered_attack(&audio, &non_finite, 1_000, 0.05, 0.01, 0.01),
            Err(RenderedAttackEvidenceErrorV1::NonFiniteAudio)
        );
        assert_eq!(
            measure_rendered_attack(&audio, &audio, 1_000, 0.005, 0.01, 0.01),
            Err(RenderedAttackEvidenceErrorV1::WindowOutOfBounds)
        );
    }
}
